# 압뒤 슬라이스 3ch 저장 전처리!
def preprocess_withLabel_3chNPY(x_list, y_list, pathToSave):
    # Image
    for i in range(len(x_list)):
        print(i, x_list[i])
        image = data_load_nii(x_list[i])
        label = data_load_nii(y_list[i])
        assert image.shape == label.shape

        for idx in range(1,len(label)-1):    
            if (np.any(label[idx])):

                slice_3_image = image[idx-1:idx+2]
                slice_3_image = np.moveaxis(slice_3_image,0,-1)                
                np.save(pathToSave + x_list[i].split('/')[-1].split('.')[0] + "_" + str(idx) + '_image_3ch.npy', slice_3_image)

                slice_3_label = label[idx-1:idx+2]
                slice_3_label = np.moveaxis(slice_3_label,0,-1)                 
                np.save(pathToSave + x_list[i].split('/')[-1].split('.')[0] + "_" + str(idx) + '_label_3ch.npy', slice_3_label)
                
def label_TPFPFN(pred,gt):
    """
    input 2d binary array torch
    TP : 1
    FN : 2
    FP : 3
    """
    
    pred_ = pred.clone()
    gt_   = gt.clone()
    gt_[gt_!=0] = 5
    pred_[pred_!=0] = 4
    result = pred_ + gt_    
       
    result[result==9] = 1    
    result[result==4] = 2    
    result[result==5] = 3  
    
    result[result==3]=0
    return result
