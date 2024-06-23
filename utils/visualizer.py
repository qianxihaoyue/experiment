

class Visualizer:



    def save_images_mask_and_predict(self, samples,paths):
        images = samples['image']
        labels = samples['label']
        predicts=samples['predict']
        assert images.shape == labels.shape==predicts.shape





