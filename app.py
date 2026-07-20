import os,re,json,subprocess,tempfile,math,hashlib
from pathlib import Path
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI

st.set_page_config(page_title='Video → AI Skill',page_icon='🧠',layout='wide')
DB=Path(os.getenv('SKILLS_DIR','skills')); DB.mkdir(exist_ok=True)
CHAT_MODEL=os.getenv('OPENAI_MODEL','gpt-4o-mini')
EMBED_MODEL=os.getenv('EMBED_MODEL','text-embedding-3-small')

def video_id(url):
    for p in [r'youtu\.be/([^?&/]+)',r'[?&]v=([^?&/]+)',r'youtube\.com/shorts/([^?&/]+)']:
        m=re.search(p,url)
        if m:return m.group(1)

def playlist_urls(url):
    if 'list=' not in url:return [url]
    try:
        p=subprocess.run(['yt-dlp','--flat-playlist','--print','%(webpage_url)s',url],check=True,capture_output=True,text=True)
        urls=[x.strip() for x in p.stdout.splitlines() if x.strip()]
        return urls or [url]
    except Exception:return [url]

def yt_transcript(url):
    v=video_id(url)
    if not v:return None
    try:
        api=YouTubeTranscriptApi()
        rows=api.fetch(v,languages=['ar','en'])
        return [{'start':float(x.start),'duration':float(x.duration),'text':x.text} for x in rows]
    except Exception:return None

def download_audio(url,d):
    out=str(Path(d)/'audio.%(ext)s')
    p=subprocess.run(['yt-dlp','-x','--audio-format','mp3','--no-playlist','-o',out,url],capture_output=True,text=True)
    if p.returncode!=0:raise RuntimeError('تعذر تنزيل الصوت: '+p.stderr[-600:])
    fs=list(Path(d).glob('audio.*'))
    if not fs:raise RuntimeError('ملف الصوت لم يتم إنشاؤه')
    return fs[0]

def transcribe(path,client):
    if path.stat().st_size>24*1024*1024:raise RuntimeError('الصوت أكبر من حد التفريغ المباشر. قص الفيديو أو استخدم captions في النسخة الحالية.')
    with open(path,'rb') as f:t=client.audio.transcriptions.create(model='whisper-1',file=f,response_format='verbose_json')
    segs=getattr(t,'segments',None)
    if segs:return [{'start':float(s.start),'duration':float(s.end-s.start),'text':s.text} for s in segs]
    return [{'start':0.0,'duration':0.0,'text':t.text}]

def stamp(sec):
    sec=max(0,int(sec));return f'{sec//3600:02}:{(sec%3600)//60:02}:{sec%60:02}'

def chunks(segs,max_chars=3200,overlap=2):
    out=[];cur=[];n=0
    for s in segs:
        if cur and n+len(s['text'])>max_chars:
            out.append({'start':cur[0]['start'],'end':cur[-1]['start']+cur[-1]['duration'],'text':' '.join(x['text'] for x in cur)})
            cur=cur[-overlap:];n=sum(len(x['text']) for x in cur)
        cur.append(s);n+=len(s['text'])
    if cur:out.append({'start':cur[0]['start'],'end':cur[-1]['start']+cur[-1]['duration'],'text':' '.join(x['text'] for x in cur)})
    return out

def embeds(c,texts):
    ans=[]
    for i in range(0,len(texts),100):ans += [x.embedding for x in c.embeddings.create(model=EMBED_MODEL,input=texts[i:i+100]).data]
    return ans

def cos(a,b):
    return sum(x*y for x,y in zip(a,b))/(math.sqrt(sum(x*x for x in a))*math.sqrt(sum(x*x for x in b))+1e-12)

def safe_slug(s):
    x=re.sub(r'[^A-Za-z0-9_-]+','_',s).strip('_')[:70]
    return x or hashlib.sha1(s.encode()).hexdigest()[:12]

def extract_one(url,client):
    segs=yt_transcript(url)
    method='captions'
    if not segs:
        method='whisper'
        with tempfile.TemporaryDirectory() as d:segs=transcribe(download_audio(url,d),client)
    return segs,method

def build(client,urls,name,progress=None):
    all_chunks=[];methods=[]
    for idx,url in enumerate(urls):
        if progress:progress.write(f'معالجة فيديو {idx+1} من {len(urls)}')
        segs,method=extract_one(url,client);methods.append(method)
        cs=chunks(segs)
        for x in cs:x['source_url']=url;x['video_index']=idx+1
        all_chunks+=cs
    if not all_chunks:raise RuntimeError('لم يتم استخراج أي محتوى')
    vecs=embeds(client,[x['text'] for x in all_chunks])
    for x,v in zip(all_chunks,vecs):x['embedding']=v
    sample='\n\n'.join(f"[فيديو {x['video_index']} {stamp(x['start'])}] {x['text']}" for x in all_chunks[:30])
    prompt='''حوّل محتوى كورس/فيديو إلى بطاقة معرفة دقيقة. لا تخترع. أعد JSON فقط بالمفاتيح: title, summary, topics, key_principles, procedures, terminology, suggested_questions. suggested_questions قائمة أسئلة مفيدة يمكن سؤال الخبير عنها. النص:\n'''+sample
    r=client.chat.completions.create(model=CHAT_MODEL,messages=[{'role':'user','content':prompt}],response_format={'type':'json_object'})
    meta=json.loads(r.choices[0].message.content)
    obj={'version':2,'name':name or meta.get('title','Skill'),'sources':urls,'methods':methods,'meta':meta,'chunks':all_chunks}
    slug=safe_slug(name or meta.get('title','skill'))
    (DB/f'{slug}.json').write_text(json.dumps(obj,ensure_ascii=False),encoding='utf-8')
    return slug,obj

def retrieve(client,skill,q,k=8):
    qv=embeds(client,[q])[0]
    ranked=sorted(skill['chunks'],key=lambda x:cos(qv,x['embedding']),reverse=True)
    return ranked[:k]

def answer(client,skill,q,history):
    top=retrieve(client,skill,q)
    ctx='\n\n'.join(f"[SOURCE video={x.get('video_index',1)} time={stamp(x['start'])}-{stamp(x['end'])}] {x['text']}" for x in top)
    hist='\n'.join(f"{m['role']}: {m['content']}" for m in history[-6:])
    prompt=f'''أنت خبير AI تم تدريبه على المصادر الموجودة في السياق فقط. تحدث بالعربي العامية المصرية إلا لو المستخدم طلب غير كده.
قواعد صارمة:
1. أي ادعاء عن رأي أو طريقة المدرس لازم يكون مدعوم بالسياق.
2. لو الإجابة مش موجودة بشكل كافي قل بوضوح إنها مش متغطية في المصادر.
3. ممكن تعمل استنتاج منطقي، لكن سمّه صراحة "استنتاج من المحتوى".
4. أعط إجابة عملية ومباشرة، واذكر المصادر بصيغة: [فيديو 2 - 00:14:20].
5. لا تدّعي إنك شاهدت أجزاء غير موجودة في السياق.
المحادثة السابقة:\n{hist}\nالسؤال: {q}\nالسياق:\n{ctx}'''
    r=client.chat.completions.create(model=CHAT_MODEL,messages=[{'role':'user','content':prompt}])
    return r.choices[0].message.content,top

def clean_export(skill):
    x=json.loads(json.dumps(skill))
    for c in x['chunks']:c.pop('embedding',None)
    return x

st.title('🧠 Video → AI Skill')
st.caption('حوّل فيديو أو Playlist إلى خبير AI ترجع تسأله بعدين من نفس المحتوى.')
with st.sidebar:
    st.header('الإعدادات')
    key=st.text_input('OpenAI API Key',type='password',value=os.getenv('OPENAI_API_KEY',''))
    st.caption('المفتاح لا يتم حفظه داخل ملفات الـSkill.')
    existing=list(DB.glob('*.json'))
    st.metric('Skills محفوظة',len(existing))
if not key:
    st.info('حط OpenAI API Key من الشريط الجانبي عشان تبدأ.');st.stop()
client=OpenAI(api_key=key)
create,chat,library=st.tabs(['➕ إنشاء Skill','💬 اسأل Skill','📚 المكتبة'])
with create:
    st.subheader('مصدر المعرفة')
    url=st.text_input('رابط YouTube / Playlist / رابط يدعمه yt-dlp')
    name=st.text_input('اسم الـSkill (اختياري)')
    playlist=st.checkbox('لو الرابط Playlist: استورد كل الفيديوهات',value=True)
    if st.button('إنشاء الـSkill',type='primary',use_container_width=True) and url:
        try:
            urls=playlist_urls(url) if playlist else [url]
            if len(urls)>100:raise RuntimeError('الـPlaylist أكبر من 100 فيديو. قسمها لأجزاء في النسخة الحالية.')
            with st.status('جاري بناء الـSkill...',expanded=True) as status:
                st.write(f'تم العثور على {len(urls)} فيديو/مصدر')
                slug,obj=build(client,urls,name,st)
                status.update(label='الـSkill جاهزة للاستخدام',state='complete')
            st.success(obj['name']);st.write(obj['meta'].get('summary',''))
            st.write('**موضوعات:**', '، '.join(obj['meta'].get('topics',[])))
            st.download_button('تصدير Skill بدون embeddings',json.dumps(clean_export(obj),ensure_ascii=False,indent=2),f'{slug}.json',mime='application/json')
        except Exception as e:st.error(str(e))
with chat:
    files=list(DB.glob('*.json'))
    if not files:st.warning('أنشئ Skill الأول.')
    else:
        pick=st.selectbox('اختار الخبير',[x.stem for x in files],key='chat_skill')
        skill=json.loads((DB/f'{pick}.json').read_text(encoding='utf-8'))
        st.subheader(skill.get('name',pick));st.write(skill.get('meta',{}).get('summary',''))
        suggestions=skill.get('meta',{}).get('suggested_questions',[])[:4]
        if suggestions:
            st.caption('أسئلة مقترحة: '+' | '.join(suggestions))
        session_key='messages_'+pick
        if session_key not in st.session_state:st.session_state[session_key]=[]
        for m in st.session_state[session_key]:
            with st.chat_message(m['role']):st.write(m['content'])
        q=st.chat_input('اسأل الخبير عن أي حاجة في الكورس...')
        if q:
            hist=st.session_state[session_key]
            with st.chat_message('user'):st.write(q)
            ans,src=answer(client,skill,q,hist)
            hist.append({'role':'user','content':q});hist.append({'role':'assistant','content':ans})
            with st.chat_message('assistant'):st.write(ans)
            with st.expander('شوف الأدلة من المصادر'):
                for x in src:
                    st.markdown(f"**فيديو {x.get('video_index',1)} — {stamp(x['start'])} → {stamp(x['end'])}**")
                    st.caption(x.get('source_url',''));st.write(x['text'])
with library:
    files=list(DB.glob('*.json'))
    if not files:st.info('المكتبة فاضية.')
    for p in files:
        try:
            s=json.loads(p.read_text(encoding='utf-8'))
            with st.expander(s.get('name',p.stem)):
                st.write(s.get('meta',{}).get('summary',''))
                st.write(f"المصادر: {len(s.get('sources',[s.get('source')]))} | أجزاء المعرفة: {len(s.get('chunks',[]))}")
                st.download_button('تصدير',json.dumps(clean_export(s),ensure_ascii=False,indent=2),p.name,key='dl_'+p.stem)
                if st.button('حذف Skill',key='del_'+p.stem):
                    p.unlink();st.rerun()
        except Exception as e:st.error(f'{p.name}: {e}')
