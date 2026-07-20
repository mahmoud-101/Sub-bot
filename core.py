import subprocess,tempfile,math,json,os
from pathlib import Path

def split_audio(path, outdir, seconds=600):
    pattern=str(Path(outdir)/'part_%03d.mp3')
    p=subprocess.run(['ffmpeg','-y','-i',str(path),'-f','segment','-segment_time',str(seconds),'-c:a','libmp3lame','-b:a','64k',pattern],capture_output=True,text=True)
    if p.returncode!=0:raise RuntimeError('فشل تقسيم الصوت: '+p.stderr[-500:])
    return sorted(Path(outdir).glob('part_*.mp3'))

def transcribe_large(path,client):
    with tempfile.TemporaryDirectory() as d:
        parts=split_audio(path,d) if path.stat().st_size>20*1024*1024 else [path]
        result=[];offset=0.0
        for part in parts:
            with open(part,'rb') as f:t=client.audio.transcriptions.create(model='whisper-1',file=f,response_format='verbose_json')
            segs=getattr(t,'segments',None)
            if segs:
                for s in segs:result.append({'start':offset+float(s.start),'duration':float(s.end-s.start),'text':s.text})
                offset=result[-1]['start']+result[-1]['duration']
            else:
                result.append({'start':offset,'duration':0.0,'text':t.text});offset+=600
        return result

def reciprocal_rank_fusion(vector_ranked, keyword_ranked, k=60):
    scores={}
    items={}
    for ranked in (vector_ranked,keyword_ranked):
        for rank,item in enumerate(ranked):
            key=(item.get('video_index',1),item.get('start',0),item.get('text','')[:80])
            items[key]=item;scores[key]=scores.get(key,0)+1/(k+rank+1)
    return [items[x] for x in sorted(scores,key=scores.get,reverse=True)]

def keyword_rank(chunks,query):
    terms={x.lower() for x in query.split() if len(x)>2}
    return sorted(chunks,key=lambda c:sum(c['text'].lower().count(t) for t in terms),reverse=True)

def save_manifest(skill_dir,skill):
    manifest={'version':skill.get('version',3),'name':skill.get('name'),'sources':skill.get('sources',[]),'meta':skill.get('meta',{}),'chunk_count':len(skill.get('chunks',[]))}
    Path(skill_dir,'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
