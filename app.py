import os,re,json,subprocess,tempfile,math,hashlib
from pathlib import Path
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI
from core import transcribe_large,reciprocal_rank_fusion,keyword_rank

st.set_page_config(page_title='Video → AI Skill',page_icon='🧠',layout='wide')
DB=Path(os.getenv('SKILLS_DIR','skills'));DB.mkdir(exist_ok=True)
CHAT_MODEL=os.getenv('OPENAI_MODEL','gpt-4o-mini');EMBED_MODEL=os.getenv('EMBED_MODEL','text-embedding-3-small')

def video_id(u):
    for p in [r'youtu\.be/([^?&/]+)',r'[?&]v=([^?&/]+)',r'youtube\.com/shorts/([^?&/]+)']:
        m=re.search(p,u)
        if m:return m.group(1)
def playlist_urls(u):
    if 'list=' not in u:return [u]
    try:
        p=subprocess.run(['yt-dlp','--flat-playlist','--print','%(webpage_url)s',u],check=True,capture_output=True,text=True)
        return [x.strip() for x in p.stdout.splitlines() if x.strip()] or [u]
    except:return [u]
def yt_transcript(u):
    v=video_id(u)
    if not v:return None
    try:
        rows=YouTubeTranscriptApi().fetch(v,languages=['ar','en'])
        return [{'start':float(x.start),'duration':float(x.duration),'text':x.text} for x in rows]
    except:return None
def download_audio(u,d):
    out=str(Path(d)/'audio.%(ext)s');p=subprocess.run(['yt-dlp','-x','--audio-format','mp3','--audio-quality','5','--no-playlist','-o',out,u],capture_output=True,text=True)
    if p.returncode:raise RuntimeError('تعذر تنزيل الصوت: '+p.stderr[-500:])
    fs=list(Path(d).glob('audio.*'))
    if not fs:raise RuntimeError('لم يتم إنشاء ملف الصوت')
    return fs[0]
def stamp(s):
    s=max(0,int(s));return f'{s//3600:02}:{s%3600//60:02}:{s%60:02}'
def make_chunks(ss,max_chars=3000,overlap=2):
    out=[];cur=[];n=0
    for s in ss:
        if cur and n+len(s['text'])>max_chars:
            out.append({'start':cur[0]['start'],'end':cur[-1]['start']+cur[-1]['duration'],'text':' '.join(x['text'] for x in cur)})
            cur=cur[-overlap:];n=sum(len(x['text']) for x in cur)
        cur.append(s);n+=len(s['text'])
    if cur:out.append({'start':cur[0]['start'],'end':cur[-1]['start']+cur[-1]['duration'],'text':' '.join(x['text'] for x in cur)})
    return out
def embeds(c,texts):
    out=[]
    for i in range(0,len(texts),100):out += [x.embedding for x in c.embeddings.create(model=EMBED_MODEL,input=texts[i:i+100]).data]
    return out
def cos(a,b):return sum(x*y for x,y in zip(a,b))/(math.sqrt(sum(x*x for x in a))*math.sqrt(sum(x*x for x in b))+1e-12)
def slug(s):return re.sub(r'[^A-Za-z0-9_-]+','_',s).strip('_')[:70] or hashlib.sha1(s.encode()).hexdigest()[:12]
def extract_url(u,c):
    ss=yt_transcript(u)
    if ss:return ss,'captions'
    with tempfile.TemporaryDirectory() as d:return transcribe_large(download_audio(u,d),c),'whisper'
def extract_upload(f,c):
    suffix=Path(f.name).suffix or '.mp4'
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/('upload'+suffix);p.write_bytes(f.getbuffer())
        if suffix.lower() in ['.txt','.md']:
            return [{'start':0,'duration':0,'text':p.read_text(encoding='utf-8',errors='ignore')}],'text-upload'
        audio=Path(d)/'audio.mp3';r=subprocess.run(['ffmpeg','-y','-i',str(p),'-vn','-b:a','64k',str(audio)],capture_output=True,text=True)
        if r.returncode:raise RuntimeError('تعذر قراءة الملف المرفوع')
        return transcribe_large(audio,c),'upload-whisper'
def build(c,sources,name,status):
    allc=[];methods=[]
    for i,src in enumerate(sources):
        status.write(f'معالجة المصدر {i+1} من {len(sources)}')
        ss,method=(extract_upload(src,c) if hasattr(src,'getbuffer') else extract_url(src,c));methods.append(method)
        cs=make_chunks(ss)
        for x in cs:x['source_url']=getattr(src,'name',src);x['video_index']=i+1
        allc+=cs
    if not allc:raise RuntimeError('لا يوجد محتوى صالح')
    for x,v in zip(allc,embeds(c,[x['text'] for x in allc])):x['embedding']=v
    sample='\n\n'.join(f"[مصدر {x['video_index']} {stamp(x['start'])}] {x['text']}" for x in allc[:35])
    prompt='''ابنِ بطاقة معرفة دقيقة من المادة. لا تخترع. JSON فقط: title, summary, topics, key_principles, procedures, terminology, suggested_questions. النص:\n'''+sample
    r=c.chat.completions.create(model=CHAT_MODEL,messages=[{'role':'user','content':prompt}],response_format={'type':'json_object'})
    meta=json.loads(r.choices[0].message.content);obj={'version':3,'name':name or meta.get('title','Skill'),'sources':[getattr(x,'name',x) for x in sources],'methods':methods,'meta':meta,'chunks':allc}
    sid=slug(obj['name']);(DB/f'{sid}.json').write_text(json.dumps(obj,ensure_ascii=False),encoding='utf-8');return sid,obj
def retrieve(c,s,q,k=8):
    qv=embeds(c,[q])[0];vr=sorted(s['chunks'],key=lambda x:cos(qv,x['embedding']),reverse=True)[:20];kr=keyword_rank(s['chunks'],q)[:20]
    return reciprocal_rank_fusion(vr,kr)[:k]
def answer(c,s,q,h):
    top=retrieve(c,s,q);ctx='\n\n'.join(f"[SOURCE {x.get('video_index',1)} {stamp(x['start'])}-{stamp(x['end'])}] {x['text']}" for x in top);hist='\n'.join(f"{m['role']}: {m['content']}" for m in h[-6:])
    p=f'''أنت خبير AI مبني فقط على المصادر. جاوب بالعامية المصرية بوضوح وبشكل عملي. لا تنسب للمدرس كلاماً غير موجود. لو الدليل غير كافٍ قل إن الموضوع مش متغطي بشكل كافي. لو استنتجت سمّه استنتاج من المحتوى. استشهد [مصدر 1 - 00:10:20].\nالمحادثة:{hist}\nالسؤال:{q}\nالمصادر:{ctx}'''
    r=c.chat.completions.create(model=CHAT_MODEL,messages=[{'role':'user','content':p}]);return r.choices[0].message.content,top
def clean(s):
    x=json.loads(json.dumps(s));[c.pop('embedding',None) for c in x['chunks']];return x

def source_link(x):
    u=x.get('source_url','');v=video_id(u)
    if v:return f'https://www.youtube.com/watch?v={v}&t={int(x["start"])}s'
    return u

st.title('🧠 Video → AI Skill');st.caption('حوّل فيديو، Playlist، ملف فيديو/صوت أو Transcript إلى خبير AI قابل للسؤال.')
with st.sidebar:
    key=st.text_input('OpenAI API Key',type='password',value=os.getenv('OPENAI_API_KEY',''));st.caption('المفتاح لا يُحفظ في الـSkill.');st.metric('Skills محفوظة',len(list(DB.glob('*.json'))))
if not key:st.info('حط OpenAI API Key في الشريط الجانبي.');st.stop()
client=OpenAI(api_key=key)
a,b,d=st.tabs(['➕ إنشاء','💬 محادثة','📚 المكتبة'])
with a:
    mode=st.radio('طريقة الإدخال',['رابط / Playlist','رفع ملف'],horizontal=True);name=st.text_input('اسم الـSkill (اختياري)');sources=[]
    if mode=='رابط / Playlist':
        url=st.text_input('الرابط');whole=st.checkbox('استورد Playlist كاملة لو موجودة',True)
        if url:sources=playlist_urls(url) if whole else [url]
    else:
        uploads=st.file_uploader('ارفع فيديو/صوت/Transcript',accept_multiple_files=True,type=['mp4','mp3','m4a','wav','txt','md']);sources=uploads or []
    if sources:st.caption(f'عدد المصادر الجاهزة: {len(sources)}')
    if st.button('ابنِ الـSkill',type='primary',use_container_width=True) and sources:
        try:
            if len(sources)>100:raise RuntimeError('قسم الكورس لأجزاء أقل من 100 مصدر.')
            with st.status('جاري بناء قاعدة المعرفة...',expanded=True) as status:
                sid,obj=build(client,sources,name,st);status.update(label='الـSkill جاهزة',state='complete')
            st.success(obj['name']);st.write(obj['meta'].get('summary',''));st.download_button('تصدير نسخة قابلة للنقل',json.dumps(clean(obj),ensure_ascii=False,indent=2),sid+'.json')
        except Exception as e:st.error(str(e))
with b:
    fs=list(DB.glob('*.json'))
    if not fs:st.warning('أنشئ Skill الأول.')
    else:
        pick=st.selectbox('اختار الخبير',[x.stem for x in fs]);s=json.loads((DB/f'{pick}.json').read_text(encoding='utf-8'));st.subheader(s.get('name',pick));st.write(s.get('meta',{}).get('summary',''))
        qs=s.get('meta',{}).get('suggested_questions',[])[:4]
        if qs:st.caption('جرّب تسأل: '+' | '.join(qs))
        sk='messages_'+pick
        if sk not in st.session_state:st.session_state[sk]=[]
        for m in st.session_state[sk]:
            with st.chat_message(m['role']):st.write(m['content'])
        q=st.chat_input('اسأل من محتوى الكورس...')
        if q:
            h=st.session_state[sk]
            with st.chat_message('user'):st.write(q)
            ans,src=answer(client,s,q,h);h += [{'role':'user','content':q},{'role':'assistant','content':ans}]
            with st.chat_message('assistant'):st.write(ans)
            with st.expander('الأدلة المستخدمة'):
                for x in src:
                    st.markdown(f"**مصدر {x.get('video_index',1)} — {stamp(x['start'])} → {stamp(x['end'])}**")
                    link=source_link(x)
                    if link.startswith('http'):st.markdown(f'[افتح المصدر عند التوقيت]({link})')
                    st.write(x['text'])
with d:
    fs=list(DB.glob('*.json'))
    if not fs:st.info('المكتبة فاضية.')
    for p in fs:
        s=json.loads(p.read_text(encoding='utf-8'))
        with st.expander(s.get('name',p.stem)):
            st.write(s.get('meta',{}).get('summary',''));st.caption(f"{len(s.get('sources',[]))} مصادر • {len(s.get('chunks',[]))} أجزاء معرفة")
            st.download_button('تصدير',json.dumps(clean(s),ensure_ascii=False,indent=2),p.name,key='dl'+p.stem)
            if st.button('حذف',key='del'+p.stem):p.unlink();st.rerun()
