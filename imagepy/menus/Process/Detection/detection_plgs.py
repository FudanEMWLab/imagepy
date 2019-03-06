from imagepy.core.engine import Filter
from imagepy.core.detection import detection


class Detection(Filter):
    title = 'Detection'
    note = ['all', 'auto_msk', 'auto_snap', 'not_channel']
  
    def run(self, ips, snap, img, para = None):
        print(snap.shape)
        return detection.detection_test(snap)
       
plgs = [Detection]
