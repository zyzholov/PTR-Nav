"""One publication owner and one replaceable prepared map, never a FIFO.

This owns already-computed message buffers. It grants no freshness exemption:
encoding and the final age check belong inside the supplied publish function.
"""
import threading


class BoundedEgress:
    def __init__(self,publish,on_error):
        self.publish=publish;self.on_error=on_error
        self.condition=threading.Condition();self.pending=None;self.stopping=False
        self.replaced=0
        self.thread=threading.Thread(target=self._run,name='terrain_egress')

    def start(self):self.thread.start()

    def submit(self,job):
        with self.condition:
            if self.stopping:return False
            if self.pending is not None:self.replaced+=1
            self.pending=job;self.condition.notify()
            return True

    def close(self,timeout=10.):
        with self.condition:
            self.stopping=True;self.pending=None;self.condition.notify_all()
        self.thread.join(timeout)
        return not self.thread.is_alive()

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda:self.stopping or self.pending is not None)
                if self.stopping:return
                job=self.pending;self.pending=None
            try:self.publish(job)
            except Exception as exc:self.on_error(exc)
