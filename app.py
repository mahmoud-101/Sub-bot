import os,re,json,subprocess,tempfile,math
from pathlib import Path
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI

st.set_page_config(page_title='Video → AI Skill',layout='wide')
st.title('Video → AI Skill')
st.caption('حط رابط فيديو، ابنِ منه Skill معرفية، وبعدها اسألها من محتوى الفيديو')
DB=Path('skills'); DB.mkdir(exist_ok=True)

def video_id(url):
    for p in [r'youtu\.be/([^?&/]+)',r'[?&]v=([^?&/]+)',r'youtube\.com/shorts/([^?&/]+)']:
        m=re.search(p,url)
        if m:return m.group(1)

def get_youtube_transcript(url):
    v=video_id(url)
    if not v:return None
    try:
        rows=YouTubeTranscriptApi().fetch(v,languages=['ar','en'])
        return [{'start':float(x.start),'duration':float(x.duration),'text':x.text} for x in rows]
    except Exception:return None

def download_audio(url,d):
    out=str(Path(d)/'audio.%(ext)s')
    subprocess.run(['yt-dlp','-x','--audio-format','mp3','-o',out,url],check=True,capture_output=True,text=True)
    files=list(Path(d).glob('audio.*'))
    if not files:raise RuntimeError('تعذر تنزيل الصوت من الرابط')
    return files[0]

def transcribe(path,client):
    with open(path,'rb') as f:
        t=client.audio.transcriptions.create(model='whisper-1',file=f,response_format='verbose_json')
    segs=getattr(t,'segments',None)
    if segs:return [{'start':float(s.start),'duration':float(s.end-s.start),'text':s.text} for s in segs]
    return [{'start':0.0,'duration':0.0,'text':t.text}]

def stamp(sec):
    sec=int(sec); return f'{sec//3600:02}:{(sec%3600)//60:02}:{sec%60:02}'

def make_chunks(segs,max_chars=4200):
    out=[]; cur=[]; n=0
    for s in segs:
        if cur and n+len(s['text'])>max_chars:
            out.append({'start':cur[0]['start'],'end':cur[-1]['start']+cur[-1]['duration'],'text':' '.join(x['text'] for x in cur)})
            cur=[]; n=0
        cur.append(s); n+=len(s['text'])
    if cur:out.append({'start':cur[0]['start'],'end':cur[-1]['start']+cur[-1]['duration'],'text':' '.join(x['text'] for x in cur)})
    return out

def embeddings(client,texts):
    return [x.embedding for x in client.embeddings.create(model='text-embedding-3-small',input=texts).data]

def cosine(a,b):
    return sum(x*y for x,y in zip(a,b))/(math.sqrt(sum(x*x for x in a))*math.sqrt(sum(x*x for x in b))+1e-12)

def build_skill(client,url,segs,name):
    chunks=make_chunks(segs); vecs=embeddings(client,[x['text'] for x in chunks])
    for x,v in zip(chunks,vecs):x['embedding']=v
    sample='\n\n'.join(f"[{stamp(x['start'])}-{stamp(x['end'])}] {x['text']}" for x in chunks[:15])
    prompt='''حلل مادة الفيديو التالية وابنِ بطاقة معرفة دقيقة. لا تضف معلومات غير مدعومة بالنص. أعد JSON فقط بالمفاتيح: title, summary, topics, key_principles, procedures, terminology.\nالنص:\n'''+sample
    r=client.chat.completions.create(model=os.getenv('OPENAI_MODEL','gpt-4o-mini'),messages=[{'role':'user','content':prompt}],response_format={'type':'json_object'})
    meta=json.loads(r.choices[0].message.content)
    slug=re.sub(r'[^A-Za-z0-9_-]+','_',name or meta.get('title','skill'))[:70] or 'skill'
    obj={'name':name or meta.get('title','Skill'),'source':url,'meta':meta,'chunks':chunks}
    (DB/f'{slug}.json').write_text(json.dumps(obj,ensure_ascii=False),encoding='utf-8')
    return slug,obj

def answer(client,skill,q):
    qv=embeddings(client,[q])[0]
    top=sorted(skill['chunks'],key=lambda x:cosine(qv,x['embedding']),reverse=True)[:7]
    ctx='\n\n'.join(f"[{stamp(x['start'])}-{stamp(x['end'])}] {x['text']}" for x in top)
    prompt=f'''أنت Skill معرفية مبنية حصراً على فيديو. جاوب بالعربي العامية المصرية بوضوح. لا تنسب للمحاضر معلومة غير موجودة في السياق. لو الإجابة غير موجودة قل: الموضوع ده مش متغطي بشكل كافي في الفيديو. لو عملت استنتاجاً قل بوضوح إنه استنتاج من المحتوى. اذكر التوقيتات الداعمة في النهاية.\nالسؤال: {q}\nالسياق:\n{ctx}'''
    r=client.chat.completions.create(model=os.getenv('OPENAI_MODEL','gpt-4o-mini'),messages=[{'role':'user','content':prompt}])
    return r.choices[0].message.content,top

key=st.sidebar.text_input('OpenAI API Key',type='password',value=os.getenv('OPENAI_API_KEY',''))
if not key:
    st.info('حط OpenAI API Key في الشريط الجانبي عشان تبدأ.'); st.stop()
client=OpenAI(api_key=key)
create,chat=st.tabs(['إنشاء Skill','اسأل Skill'])
with create:
    url=st.text_input('رابط الفيديو')
    name=st.text_input('اسم الـ Skill (اختياري)')
    if st.button('استخرج الفيديو وابنِ Skill',type='primary') and url:
        try:
            with st.status('جاري المعالجة...',expanded=True) as status:
                st.write('1) محاولة استخراج Transcript مباشر...')
                segs=get_youtube_transcript(url)
                if not segs:
                    st.write('2) مفيش Transcript مباشر؛ جاري تنزيل الصوت وتشغيل Whisper...')
                    with tempfile.TemporaryDirectory() as d:segs=transcribe(download_audio(url,d),client)
                st.write(f'3) تم استخراج {len(segs)} مقطع. جاري بناء قاعدة المعرفة...')
                slug,obj=build_skill(client,url,segs,name)
                status.update(label='تم إنشاء الـ Skill',state='complete')
            st.success(f"Skill جاهزة: {obj['name']}")
            st.json(obj['meta'])
            clean=[{k:v for k,v in x.items() if k!='embedding'} for x in obj['chunks']]
            st.download_button('تنزيل النص المستخرج',json.dumps(clean,ensure_ascii=False,indent=2),f'{slug}_transcript.json')
        except Exception as e:st.error(f'حصل خطأ: {e}')
with chat:
    files=list(DB.glob('*.json'))
    if not files:st.warning('أنشئ Skill الأول من التبويب الأول.')
    else:
        pick=st.selectbox('اختار Skill',[x.stem for x in files])
        skill=json.loads((DB/f'{pick}.json').read_text(encoding='utf-8'))
        st.write(skill['meta'].get('summary',''))
        q=st.chat_input('اسأل أي سؤال من الفيديو...')
        if q:
            with st.chat_message('user'):st.write(q)
            ans,src=answer(client,skill,q)
            with st.chat_message('assistant'):st.write(ans)
            with st.expander('المقاطع اللي اتبنت عليها الإجابة'):
                for x in src:
                    st.markdown(f"**{stamp(x['start'])} → {stamp(x['end'])}**")
                    st.write(x['text'])
