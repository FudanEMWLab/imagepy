from imagepy.core.engine import Filter
from imagepy import IPy
from imagepy.core.classify import classify


class Classify_Cap(Filter):
    title = 'Classify'
    note = ['all', 'auto_msk', 'auto_snap', 'not_channel']
  
    def run(self, ips, snap, img, para = None):
        print(snap.shape)
        label = classify.classify_cap(snap)
        print(label)
        IPy.alert('This cap is : {}'.format(label), title="classified result")
        
plgs = [Classify_Cap]
