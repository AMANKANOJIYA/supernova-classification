import sys, json
from confluent_kafka import TopicPartition
from fink_client.consumer import AlertConsumer
from src.serve.live_fink import FINK_SERVERS, DEFAULT_TOPICS
survey=sys.argv[1]
cfg={"username":"aman","group.id":"aman_kanojiya_dhanbad","bootstrap.servers":FINK_SERVERS[survey]}
c=AlertConsumer(topics=DEFAULT_TOPICS[survey], config=cfg, survey=survey)
kc=c._consumer
md=kc.list_topics(timeout=20)
rows=[]
for t,tm in sorted(md.topics.items()):
    if t.startswith("__"): continue
    lo=hi=0
    for p in tm.partitions:
        l,h=kc.get_watermark_offsets(TopicPartition(t,p),timeout=5); lo+=l; hi+=h
    rows.append(dict(topic=t,partitions=len(tm.partitions),low=lo,high=hi,retained=hi-lo))
    print("%-55s parts=%-3d low=%-10d high=%-10d retained=%d"%(t,len(tm.partitions),lo,hi,hi-lo), flush=True)
json.dump(rows,open(sys.argv[2],"w"),indent=1)
c.close()
