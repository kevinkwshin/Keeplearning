def sample_patches_randomly(self, num_patches, d , h , w ):

        '''
        INPUT:
        num_patches : the total number of samled patches
        d : this correspnds to the number of channels which is ,in our case, 4 MRI modalities
        h : height of the patch
        w : width of the patch
        OUTPUT:
        patches : np array containing the randomly sampled patches
        labels : np array containing the corresping target patches
        '''
        patches, labels = [], []
        count = 0

        #swap axes to make axis 0 represents the modality and axis 1 represents the slice. take the ground truth
        gt_im = np.swapaxes(self.train_im, 0, 1)[4]   

        #take flair image as mask
        msk = np.swapaxes(self.train_im, 0, 1)[0]
        #save the shape of the grounf truth to use it afterwards
        tmp_shp = gt_im.shape

        #reshape the mask and the ground truth to 1D array
        gt_im = gt_im.reshape(-1).astype(np.uint8)
        msk = msk.reshape(-1).astype(np.float32)

        # maintain list of 1D indices while discarding 0 intensities
        indices = np.squeeze(np.argwhere((msk!=-9.0) & (msk!=0.0)))
        del msk

        # shuffle the list of indices of the class
        np.random.shuffle(indices)

        #reshape gt_im
        gt_im = gt_im.reshape(tmp_shp)

        #a loop to sample the patches from the images
        i = 0
        pix = len(indices)
        while (count<num_patches) and (pix>i):
            #randomly choose an index
            ind = indices[i]
            i+= 1
            #reshape ind to 3D index
            ind = np.unravel_index(ind, tmp_shp)
            # get the patient and the slice id
            patient_id = ind[0]
            slice_idx=ind[1]
            p = ind[2:]
            #construct the patch by defining the coordinates
            p_y = (p[0] - (h)/2, p[0] + (h)/2)
            p_x = (p[1] - (w)/2, p[1] + (w)/2)
            p_x=list(map(int,p_x))
            p_y=list(map(int,p_y))
            
            #take patches from all modalities and group them together
            tmp = self.train_im[patient_id][0:4, slice_idx,p_y[0]:p_y[1], p_x[0]:p_x[1]]
            #take the coresponding label patch
            lbl=gt_im[patient_id,slice_idx,p_y[0]:p_y[1], p_x[0]:p_x[1]]

            #keep only paches that have the desired size
            if tmp.shape != (d, h, w) :
                continue
            patches.append(tmp)
            labels.append(lbl)
            count+=1
        patches = np.array(patches)
        labels=np.array(labels)
        return patches, labels
