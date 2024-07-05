

class Visualizer:

    def __init__(self):
        pass

    @staticmethod
    def save_images_mask_and_predict(self, samples,paths):
        images = samples['image']
        labels = samples['label']
        predicts=samples['predict']
        assert images.shape == labels.shape==predicts.shape

    def save_predict_and_image(self,predict,image,**kargs):
        pass








