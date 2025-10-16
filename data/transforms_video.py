import random
import numpy as np
import cv2

# Video transformation classes
class GroupMultiScaleCrop:
    def __init__(self, input_size, scales):
        self.input_size=input_size; self.scales=scales if scales is not None else [1,.875,.75,.66]
    def __call__(self, img_group):
        scale=random.choice(self.scales); new_size=int(round(self.input_size/scale)); transformed=[]
        for img in img_group:
            img_resized=cv2.resize(img,(new_size,new_size)); y1,x1=(random.randint(0,new_size-self.input_size),random.randint(0,new_size-self.input_size))
            transformed.append(img_resized[y1:y1+self.input_size,x1:x1+self.input_size,:])
        return transformed

class GroupRandomHorizontalFlip:
    def __init__(self, is_mv=False): self.is_mv = is_mv
    def __call__(self, img_group):
        if random.random()<0.5:
            flipped=[]
            for img in img_group:
                img=np.flip(img,axis=1).copy(); 
                if self.is_mv: img[:,:,0]=255-img[:,:,0]
                flipped.append(img)
            return flipped
        return img_group

class GroupScale:
    def __init__(self, new_size): self.new_size = new_size
    def __call__(self, img_group):
        return [cv2.resize(img, (self.new_size, self.new_size)) for img in img_group]

class GroupCenterCrop:
    def __init__(self, crop_size): self.crop_size = crop_size
    def __call__(self, img_group):
        cropped=[]
        for img in img_group:
            h,w,_=img.shape; y1,x1=(int(round((h-self.crop_size)/2)),int(round((w-self.crop_size)/2)))
            cropped.append(img[y1:y1+self.crop_size,x1:x1+self.crop_size,:])
        return cropped

class GroupOverSample:
    def __init__(self, crop_size, scale_size, is_mv=False):
        self.crop_size=crop_size; self.scale_size=scale_size if scale_size is not None else crop_size; self.is_mv=is_mv
        self.flip=GroupRandomHorizontalFlip(is_mv=self.is_mv)
    def __call__(self, img_group):
        img_group_scaled=[cv2.resize(img,(self.scale_size,self.scale_size)) for img in img_group]
        h,w,_=img_group_scaled[0].shape
        crop_positions=[(0,0),(0,w-self.crop_size),(h-self.crop_size,0),(h-self.crop_size,w-self.crop_size),
                          (int(round((h-self.crop_size)/2)),int(round((w-self.crop_size)/2)))]
        oversampled_imgs=[]
        for y,x in crop_positions:
            cropped_group=[img[y:y+self.crop_size,x:x+self.crop_size,:] for img in img_group_scaled]
            oversampled_imgs.append(cropped_group)
        # Flip and crop again
        img_group_flipped=self.flip(img_group_scaled)
        for y,x in crop_positions:
            cropped_group=[img[y:y+self.crop_size,x:x+self.crop_size,:] for img in img_group_flipped]
            oversampled_imgs.append(cropped_group)
        return oversampled_imgs



# Additional classes for 1-crop testing
class GroupCenterSample:
    def __init__(self, crop_size, scale_size=None):
        self.crop_size = crop_size
        # If scale_size is not provided, default to crop_size
        self.scale_size = scale_size if scale_size is not None else crop_size
        
        # Reuse existing classes
        self.scaler = GroupScale(self.scale_size)
        self.cropper = GroupCenterCrop(self.crop_size)

    def __call__(self, img_group):
        # First scale
        scaled_group = self.scaler(img_group)
        # Then center crop
        cropped_group = self.cropper(scaled_group)
        
        # GroupOverSample outputs a list of lists, to maintain data format consistency
        # we also wrap it as [cropped_group]
        return [cropped_group]