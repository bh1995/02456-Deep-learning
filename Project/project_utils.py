
########## Functions for U-Net ##########

def get_data(path, start, end, size=(128,128)):
  """ 
  function to load image data of cells and normalize the pictures for dataloader.
  """
  images = []
  annotations = []
  image_names = []
  mask_names = []
  # First get names of all images to read organized
  image_names = [ f.name for f in os.scandir(path+'/image')][start:end]
  image_names = natsorted(image_names) # Sort so that we get correct number matching between images and annotations
  mask_names = [ f.name for f in os.scandir(path+'/mask')][start:end]
  mask_names = natsorted(mask_names)
  # print('sorted_names', image_names)
  # print(mask_names)
  
  # Load images
  for image_name in image_names:
    im = io.imread(os.path.join(path+'/image', image_name))
    # im = rgb2gray(im)
    im = np.array(im, np.float32)
    # im = np.moveaxis(im, 2, -3)
    im = resize(im, size, anti_aliasing=True)
    pixels = asarray(im)
    pixels = pixels.astype('float32')
    mean, std = pixels.mean(), pixels.std()
    # print('Before normalization', 'Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
    # global standardization of pixels
    pixels = (pixels - mean) / std
    mean2, std2 = pixels.mean(), pixels.std()
    # assert [np.isclose([mean2, std2], [0, 1.0], atol=0.0001)] == [ True, True]
    pixels = np.moveaxis(pixels, 2, -3) # move channels to last i.e: [C,W,H]
    # print('images', pixels.shape)
    images.append(pixels)

  # Load masks
  for image_name in mask_names:
    an = io.imread(os.path.join(path+'/mask', image_name))
    # an = rgb2gray(an)
    an = np.array(an, np.float32)
    # an = np.moveaxis(an, 2, -3)
    an = resize(an, size, anti_aliasing=True)
    pixels = asarray(an)
    pixels = pixels.astype('float32')
    mean, std = pixels.mean(), pixels.std()
    # print('Before normalization', 'Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
    # global standardization of pixels
    pixels = (pixels - mean) / std
    mean2, std2 = pixels.mean(), pixels.std()
    # assert [np.isclose([mean2, std2], [0, 1.0], atol=0.0001)] == [ True, True]
    # assert [img.shape==pixels.shape] == [True]
    # print('annotations', pixels.shape)
    annotations.append(pixels)

  X = images
  Y = annotations
  del images
  del annotations
  print(image_names)
  # print(anno_names[0:10])
  # print(im_names[0:10])
  return (X, Y)

def train(model, opt, loss_fn, epochs, data_loader, print_status):

    loss_ls = []
    epoch_ls = []
    for epoch in range(epochs):
        avg_loss = 0
        model.train() 

        b=0
        for X_batch, Y_batch in data_loader:
            
            
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
         
            # set parameter gradients to zero
            opt.zero_grad()
            # print(input_size)
            # forward pass
            Y_pred = model(X_batch)
            
            """
            if (epoch % 10 ==0):
              plt.figure(figsize=(5,5))
              plt.imshow(Y_pred[-1,0,:,:].detach().numpy( ))
            """
            # print('Y_pred shape', Y_pred.shape)
            # print('Y_batch shape before', Y_batch.shape)
            Y_batch = Y_batch.unsqueeze(1)
            Y_batch[1] = Y_pred[1]
            loss = loss_fn(Y_pred, Y_batch)  # compute loss
            loss.backward()  # backward-pass to compute gradients
            opt.step()  # update weights

            # Compute average epoch loss
            avg_loss += loss / len(data_loader)
            #print(b)
            b=b+1
            # print(loss)
        
        """
        if print_status:
            print(f"Loss in epoch {epoch} was {avg_loss}")
        """
        loss_ls.append(avg_loss)
        epoch_ls.append(epoch)
        # Delete unnecessary tensors
        Y_batch[5:] = 0
        # show intermediate results
        model.eval()  # testing mode
        Y_hat = F.sigmoid(model(X_batch.to(device))).detach().cpu()
        # del X_batch
        Y_hat[5:, 0] = 0
        clear_output(wait=True)

        # plt.subplots_adjust(bottom=1, top=2, hspace=0.2)
        for k in range(4):
            plt.subplot(3, 4, k+1)
            Y_batch2 = Variable(Y_batch[k,0,:,:], requires_grad=False)
            plt.imshow(Y_batch2.cpu().numpy(), cmap='Greys')
            # plt.imshow(X_batch[k,0,:,:].cpu().numpy( ))
            # plt.imshow(Y_batch[k].cpu().numpy( ))
            plt.title('Real')
            plt.axis('off')

            plt.subplot(3, 4, k+5)
            plt.imshow(Y_hat[k, 0], cmap='Greys')
            # plt.imshow(Y_hat[k, 0])
            plt.title('Output')
            plt.axis('off')


          
        plt.suptitle('%d / %d - loss: %f' % (epoch+1, epochs, avg_loss))
        plt.show()
        plt.plot(epoch_ls, loss_ls, label='traning loss')
        plt.legend()
        plt.xlabel('Epoch'), plt.ylabel('Loss')
        plt.show()

    return model

def view_mask(targets, output, n=2, cmap='Greys'):
    figure = plt.figure(figsize=(15,10))
    for i in range(n):
      # plot target (true) masks
      target_im = targets[i].cpu().detach().numpy()
      target_im[target_im>0.5] = 1
      target_im[target_im<0.5] = 0
      ax = figure.add_subplot(2,2, i+1)
      ax.imshow(target_im, cmap=cmap)
      # Plot output (predicted) masks
      output_im = output[i][0, :, :].cpu().detach().numpy()
      output_im[output_im>0.5] = 1
      output_im[output_im<0.5] = 0
      ax = figure.add_subplot(2,2, i+3)
      ax.imshow(output_im, cmap=cmap)

def IoU(y_real, y_pred):
  # Intersection over Union loss function
  intersection = y_real*y_pred
  #not_real = 1 - y_real
  #union = y_real + (not_real*y_pred)
  union = (y_real+y_pred)-(y_real*y_pred)
  return np.sum(intersection)/np.sum(union)

def dice_coef(y_real, y_pred, smooth=1):
  intersection = y_real*y_pred
  union = (y_real+y_pred)-(y_real*y_pred)
  return np.mean((2*intersection+smooth)/(union+smooth))

# def get_confusion_matrix_elements(groundtruth_list, predicted_list):
#     """returns confusion matrix elements i.e TN, FP, FN, TP as floats
# 	See example code for helper function definitions
#     """
#     tn, fp, fn, tp = sklearn.metrics.confusion_matrix(groundtruth_list.argmax(axis=1), predicted_list.argmax(axis=1)).ravel()
#     tn, fp, fn, tp = np.float64(tn), np.float64(fp), np.float64(fn), np.float64(tp)

#     return tn, fp, fn, tp

def confusion_matrix(y_true, y_pred):
    y_true= y_true.flatten()
    y_pred = y_pred.flatten()*2
    cm = y_true+y_pred
    cm = np.bincount(cm, minlength=4)
    tn, fp, fn, tp = cm
    return tp, fp, tn, fn

def get_f1_score(y_true, y_pred):
    """Return f1 score covering edge cases"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)
    f1_score = (2 * tp) / ((2 * tp) + fp + fn)

    return f1_score     

########## Functions for Mask R-CNN ##########

def view(images,labels,n=2,std=1,mean=0):
    figure = plt.figure(figsize=(15,10))
    images=list(images)
    labels=list(labels)
    for i in range(n):
        out=torchvision.utils.make_grid(images[i])
        inp=out.cpu().numpy().transpose((1,2,0))
        inp=np.array(std)*inp+np.array(mean)
        inp=np.clip(inp,0,1)  
        ax = figure.add_subplot(2,2, i + 1)
        ax.imshow(images[i].cpu().numpy().transpose((1,2,0)))
        l=labels[i]['boxes'].cpu().numpy()
        l[:,2]=l[:,2]-l[:,0]
        l[:,3]=l[:,3]-l[:,1]
        for j in range(len(l)):
            ax.add_patch(patches.Rectangle((l[j][0],l[j][1]),l[j][2],l[j][3],linewidth=1.5,edgecolor='r',facecolor='none')) 

def latest_model():
  sp = '/content/drive/MyDrive/warwick_qu_dataset_released_2016_07_08/Warwick QU Dataset (Released 2016_07_08)/trained_models'
  largest = 0
  model_names = os.listdir(os.path.join(sp))
  for i in model_names:
    nr = int(list(filter(str.isdigit, i))[0])
    if nr>largest:
      largest = nr
  return largest

def view_mask2(targets, output, n=2, cmap='Greys'):
    figure = plt.figure(figsize=(15,10))
    for i in range(n):
      # plot target (true) masks
      target_im = targets[i]['masks'][0].cpu().detach().numpy()
      for k in range(len(targets[i]['masks'])):
        target_im2 = targets[i]['masks'][k].cpu().detach().numpy()
        target_im2[target_im2>0.5] = 1
        target_im2[target_im2<0.5] = 0
        target_im = target_im+target_im2

      target_im[target_im>0.5] = 1
      target_im[target_im<0.5] = 0
      ax = figure.add_subplot(2,2, i+1)
      ax.imshow(target_im, cmap=cmap)
      # Plot output (predicted) masks
      output_im = output[i]['masks'][0][0, :, :].cpu().detach().numpy()
      for k in range(len(output[i]['masks'])):
        output_im2 = output[i]['masks'][k][0, :, :].cpu().detach().numpy()
        output_im2[output_im2>0.5] = 1
        output_im2[output_im2<0.5] = 0
        output_im = output_im+output_im2

      output_im[output_im>0.5] = 1
      output_im[output_im<0.5] = 0
      ax = figure.add_subplot(2,2, i+3)
      ax.imshow(output_im, cmap=cmap)



