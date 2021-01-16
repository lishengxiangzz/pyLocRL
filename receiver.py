
class Receiver(object):
    def __init__(self, pos, is_center=False):
        """
        The receiver in passive location system
        :param pos: the location of the receiver, [x,y]
        :param is_center: is or not the center receiver of the passive location system
        """
        self.x = pos[0]
        self.y = pos[1]
        self.pos = pos
        self.is_center = is_center
        self.signal = []
        self.SNR = []
