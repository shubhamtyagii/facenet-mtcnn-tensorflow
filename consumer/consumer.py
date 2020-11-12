import cv2
class Consumer:
    
    def __init__(self,video_path,skip_frame_interval = 0):
        self.cap = cv2.VideoCapture(video_path)
        self.skip_frame_interval = skip_frame_interval
    def get_frame(self):
        c = 0
        while(c<self.skip_frame_interval):
            ret,_ = self.cap.read()
            c+=1
            if(not ret):
                return False,None

        return self.cap.read()


    def release(self):
        self.cap.release()