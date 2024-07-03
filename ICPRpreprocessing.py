import nibabel as nib
import os
import scipy
import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
import random 
import cv2
import numpy
from skimage.filters import threshold_otsu
import re

media = 0
des = 0
import torch
from torch.utils.data import DataLoader

def calc_mean_std(data_dl: DataLoader):
    total_sum = 0
    total_squared_sum = 0
    num_batches = 0
    num_pixels = 0

    for images, _ in data_dl:
        # Asegúrate de que las imágenes están en el tipo de dato correcto
        images = images.to(torch.float32)
        
        # Calcular el número total de píxeles en el batch actual
        batch_pixels = images.size(0) * images.size(1) * images.size(2) * images.size(3)
        
        # Acumular la suma y el cuadrado de la suma
        total_sum += torch.sum(images)
        total_squared_sum += torch.sum(images ** 2)
        
        num_batches += 1
        num_pixels += batch_pixels

    # Calcular la media y la desviación estándar
    mean = total_sum / num_pixels
    std = torch.sqrt((total_squared_sum / num_pixels) - mean ** 2)
    
    return mean, std

 

def createDictionary(path:str, DICT: dict):
    subdirectories= os.listdir(path)
    for x in subdirectories:
        for y in os.listdir(os.path.join(path, x)):
            t1_file = f"{x}_{y}_T1.nii.gz"
            t2_file = f"{x}_{y}_T2.nii.gz"
            flair_file = f"{x}_{y}_FLAIR.nii.gz"
            mask_file = f"{x}_{y}_MASK.nii.gz"
            DICT["mask"].append(os.path.join(path,x,y,mask_file))
            DICT["pathsT1"].append(os.path.join(path,x, y,t1_file))
            DICT["pathsFLAIR"].append(os.path.join(path,x, y,flair_file))
            DICT["pathsT2"].append(os.path.join(path,x, y,t2_file))
            
def createDictionaryTEST(path:str, DICT: dict):
    subdirectories= os.listdir(path)
    for x in subdirectories:
        t1_file = f"{x}_T1.nii.gz"
        t2_file = f"{x}_T2.nii.gz"
        flair_file = f"{x}_FLAIR.nii.gz"
        mask_file = f"{x}_MASK.nii.gz"
        DICT["mask"].append(os.path.join(path,x,mask_file))
        DICT["pathsT1"].append(os.path.join(path,x,t1_file))
        DICT["pathsFLAIR"].append(os.path.join(path,x,flair_file))
        DICT["pathsT2"].append(os.path.join(path,x,t2_file))

def divideDataset(dictio: dict, per=0.8):
    num_values=len(dictio["pathsFLAIR"])
    idx= list(range(num_values))
    random.shuffle(idx)
    part= int(num_values*per)
    train ={k: [v[i] for i in idx[:part]] for k, v in dictio.items()}
    val ={k: [v[i] for i in idx[part:]] for k, v in dictio.items()}
    return train,val

def getIDs(dictio:dict):
    lista =["P2_T1","P2_T4","P3_T3","P4_T2","P5_T2","P8_T1","P9_T1","P9_T2","P10_T1","P10_T2","P11_T2",
            "P20_T3","P25_T1","P36_T1","P40_T1","P45_T1","P46_T1","P51_T2","P52_T2"]
    val={k:[] for k in dictio.keys()}
    train={k:[] for k in dictio.keys()}
    for x,i in enumerate(dictio["pathsFLAIR"]):

        filename = i.split("\\")[-1]
        identifier = filename.split('_')[0] + "_" + filename.split('_')[1]
        if identifier in lista:
            for key in dictio.keys():
                val[key].append(dictio[key][x])
        else:
            for key in dictio.keys():
                train[key].append(dictio[key][x])
    return train, val        

def add_transformation(image,final_size:int,options:bool, isMask:bool):
    if(not torch.is_tensor(image)):
        image= transforms.ToTensor()(image)
    # IF ONLY RESIZE
    if(options): 
        if(isMask):resize=transforms.Resize((final_size,final_size), antialias=True,interpolation= transforms.InterpolationMode.NEAREST)
        else:resize = transforms.Resize((final_size,final_size), antialias=True)
        image=resize(image)
        
    # IF CROP 
    # ELSE PAD
    else:
        if (image.shape[1] < final_size or image.shape[2] < final_size):
            pad_width = max(final_size - image.shape[1], 0)
            pad_height = max(final_size -  image.shape[2], 0)    
            total=(pad_width // 2, pad_height // 2, pad_width - pad_width // 2, pad_height - pad_height // 2) 
            padding=transforms.Pad(total, fill=0)
            image=padding(image)
        
        if(image.shape[1]>final_size or image.shape[2]>final_size):
            crop = transforms.CenterCrop((final_size,final_size))
            image=crop(image)

        

    return image

def add_normalization(image,n:int):
    match n:
        case 1:
            image= transforms.ToTensor()(image)
            return gaussian_normalization(image)
        case 2:
            image= transforms.ToTensor()(image)
            return minmax_normalization(image)
        case 3:
            mask= brainMask(image)
            image= transforms.ToTensor()(image)
            mask= transforms.ToTensor()(mask)
            return gaussian_normalizationFILL(image, mask)
        case 4:
            mask= brainMask(image)
            image= transforms.ToTensor()(image)
            mask= transforms.ToTensor()(mask)
            return minmax_normalizationFILL(image, mask)
        case 5: 
            mask= brainfilling(brainMask(image))
            image= transforms.ToTensor()(image)
            mask= transforms.ToTensor()(mask)
            return gaussian_normalizationFILL(image, mask)
        case 6:    
            mask= brainfilling(brainMask(image))
            image= transforms.ToTensor()(image)
            mask= transforms.ToTensor()(mask)
            return minmax_normalizationFILL(image, mask)
        


#def removeSkull(image)     
    # IF REMOVE SKULL 
    ################



def brainMask(image):
# Aislar cerebro
    thr=threshold_otsu(image)
    return image > thr

# def brainfilling(image):   
#    image_np = image.cpu().numpy()   
#    image_f=scipy.ndimage.morphology.binary_fill_holes(image_np) 
#    return torch.tensor(image_f, dtype=torch.bool)

def brainfilling(image):
    return scipy.ndimage.morphology.binary_fill_holes(image) 


def gaussian_normalizationFILL(image, brain):
    mean= torch.mean(image[brain==1])
    std=torch.std(image[brain==1])
    norm =(image-mean)/std
    return norm

def minmax_normalizationFILL(image,brain):
    if image[brain==1].numel()==0:
        min = float('inf')
        max = float('inf')
    else:    
        min= torch.min(image[brain==1])
        max= torch.max(image[brain==1])
   
    norm = (image - min) / (max - min)
    return norm

def gaussian_normalization(image):
    norm =(image-media)/des
    return norm

def minmax_normalization(image):
    min= torch.min(image)
    max= torch.max(image)
    norm = (image - min) / (max - min)
    return norm


def dataAugmentation(image,imageMask, deg,sca=None,she=None):
    if(not torch.is_tensor(image)):
        image= transforms.ToTensor()(image)
    if(not torch.is_tensor(imageMask)) :   
        imageMask= transforms.ToTensor()(imageMask)
    transform_params = transforms.RandomAffine.get_params(degrees=deg, translate=(0,0), scale_ranges=sca, shears=she, img_size=image.size())    
    image= F.affine(image, *transform_params)
    imageMask = F.affine(imageMask, *transform_params)
    return image,imageMask


#ONLY RESIZE OR CROP
def transform_setter(size: int, hasResize:True):
    transform=transforms.Compose([
        lambda x: add_transformation(x, size,hasResize, False),
    ])

    transform_label=transforms.Compose([
        lambda x: add_transformation(x, size,hasResize, True),
    ])
    return transform, transform_label

#RESIZE/CROP AND NORMALIZATION
def transform_normalization(size: int, hasResize:True, n:0):
    transform=transforms.Compose([
        lambda x: add_transformation(add_normalization(x,n), size,hasResize, False),
    ])

    transform_label=transforms.Compose([
        lambda x: add_transformation(x, size,hasResize, True)
    ])
    return transform, transform_label
    
class Slices(Dataset):
    def __init__(self, images, labels, transform, transform_label, slices_deletion):
        super().__init__()
        self.paths = images
        self.labels = labels
        self.slices_deletion=slices_deletion
        self.images, self.masks= self.__totalimages__()
        self.len = len(self.images)
        self.transform = transform
        self.transform_label = transform_label
        
    def __len__(self): 
        return self.len
    
    def __totalimages__(self):
        images = []
        masks = []
        for path, label in zip(self.paths, self.labels):
            img = nib.load(path)
            image= img.get_fdata() 

            lab = nib.load(label)
            label_img =  lab.get_fdata()
            label_img[label_img==2]=0

            n_slices=image.shape[2]
            if(self.slices_deletion):
                for ii in range(0,n_slices):
                    im = image[:, :, ii]
                    lab=label_img[:, :, ii]
                    #eliminar slices en los que no haya que segmentar nada también
                    if(cv2.countNonZero(im)!=0):
                        print(ii)
                        images.append(im)
                        masks.append(lab)           
            else:
                for ii in range(0,n_slices):
                    im = image[:, :, ii]
                    lab=label_img[:, :, ii]
                    images.append(im)
                    masks.append(lab)

        return images, masks            

    def __getitem__(self, index):
            image=self.images[index]
            mask= self.masks[index]
            image = self.transform(image)
            label_img = self.transform_label(mask)    
            return image,label_img  

    

def dataLoaders(path:str,train:dict, val:dict,transform, transform_label, isShuffled=False, size=30,slices_deletion=False):    
    train_data = Slices(train.get(path), train.get("mask"), transform, transform_label,slices_deletion)
    print(train_data.len)
    val_data = Slices(val.get(path), val.get("mask"), transform, transform_label,slices_deletion)
    print(val_data.len)
    train_dl = DataLoader(train_data, batch_size=size, shuffle=isShuffled)
    val_dl = DataLoader(val_data, batch_size=size, shuffle=isShuffled)
    return train_data, val_data, train_dl,val_dl 

class Concatenate(Dataset):
    def __init__(self, flair, t1, t2,labels, transform, transform_label):
        super().__init__()
        self.flair = flair
        self.t1 = t1
        self.t2 = t2
        self.labels = labels
        self.transform = transform
        self.transform_label = transform_label
        self.images, self.masks= self.__totalimages__()
        self.len = len(self.images)
           
    def __len__(self): 
        return self.len
    
    def __totalimages__(self):
        images = []
        masks = []
        for flair,t1, label in zip(self.flair,self.t1, self.t2, self.labels):
            flair = nib.load(flair)
            flair_im= flair.get_fdata() 

            t1=nib.load(t1)
            t1_im=t1.get_fdata()

            t2 = nib.load(t2)
            t2_im= t2.get_fdata()


            lab = nib.load(label)
            label_img =  lab.get_fdata()
            

            n_slices=t1_im.shape[2]
            lim_inf= int(n_slices*0.1)
            lim_sup=int(n_slices*0.9)
            for ii in range(0,n_slices):
                t = t1_im[:, :, ii]
                t2 = t2_im[:,:, ii]
                fl=flair_im[:,:,ii]
                lab=label_img[:, :, ii]
                if(ii>lim_inf and ii<lim_sup or cv2.countNonZero(lab)!=0):
                    t= self.transform(t)
                    t2 = self.transform(t2)
                    fl=self.transform(fl)
                    lab= self.transform_label(lab)
                    im= numpy.concatenate((t,t2,fl), axis=2)
                    images.append(im)
                    masks.append(lab)           
        return images, masks            

    def __getitem__(self, index):
        image=self.images[index]
        mask= self.masks[index]  
        return image,mask       

def dataLoadersConcatenate(flair:str,t1:str,t2:str, train:dict, val:dict,transform, transform_label, isShuffled=False, size=30):    
    train_data = Concatenate(train.get(flair),train.get(t1),train.get(t2), train.get("mask"), transform, transform_label)
    print(train_data.len)
    val_data = Concatenate(val.get(flair),val.get(t1), val.get(t2),val.get("mask"), transform, transform_label)
    print(val_data.len)
    train_dl = DataLoader(train_data, batch_size=size, shuffle=isShuffled)
    val_dl = DataLoader(val_data, batch_size=size, shuffle=isShuffled)
    return train_data, val_data, train_dl,val_dl 

def generateAugmentation(trainData:Concatenate, size):
    images=[]
    masks =[]
    for _ in range(0,size):
        item= random.randint(0,trainData.__len__())
        image, mask=trainData.__getitem__(item)
        n = random.randint(1,3)
        if(n==1):
            image,mask=dataAugmentation(image,mask,(-15,15),None,None)
        elif(n==2):
            image,mask=dataAugmentation(image,mask,(0,0),(0.9,1.1),None)
        elif(n==3):  
            image,mask=dataAugmentation(image,mask,(0,0),None,(-18,18))
        images.append(image)
        masks.append(mask) 
    return images, mask             

class Augmentation(Dataset):
    def __init__(self, imagesFlair,labels, transform, transformLabel,size):
        super().__init__()
        self.imagesFlair, self.labels = self.readImages(imagesFlair, labels)
        self.transform = transform
        self.transform_label = transformLabel
        self.size = size
        self.len = len(self.imagesFlair)
        self.images, self.masks = self.generateAugmentation()
        
        
    def __len__(self): 
        return self.len
    
    def readImages(self, paths, pathsM):
        images =[]
        masks = []
        for flair, label in zip(paths, pathsM):
            flair = nib.load(flair)
            flair_im= flair.get_fdata(dtype=numpy.float32)
            n_slices=flair_im.shape[2]

            lab = nib.load(label)
            label_img =  lab.get_fdata(dtype=numpy.float32)
            label_img[label_img==2]=0
            for ii in range(0,n_slices):
                fl=flair_im[:,:,ii]
                l = label_img[:,:,ii]
                if (cv2.countNonZero(fl)!=0):
                    images.append(fl)
                    masks. append(l)
        return images, masks
          


    def  generateAugmentation(self):
        imagesAUX=[]
        masksAUX =[]
        for _ in range(0,self.size):
            item = random.randint(0,self.len-1)
            imageF = self.imagesFlair[item]
            mask = self.masks[item]
            n = random.randint(1,3)
            if(n==1):
                fl,mask=dataAugmentation(imageF, mask,(-15,15),None,None)
            elif(n==2):
                fl,mask=dataAugmentation(imageF, mask,(0,0),(0.9,1.1),None)
            elif(n==3):  
                fl,mask=dataAugmentation(imageF,mask,(0,0),None,(-18,18))
            imagesAUX.append(fl)
            masksAUX.append(mask)           
        return imagesAUX, masksAUX      

    def __getitem__(self, index):
        image=self.images[index]
        mask= self.masks[index] 
        return image,mask   
     
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 0.0

    def forward(self, y_pred, y_true):
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc    