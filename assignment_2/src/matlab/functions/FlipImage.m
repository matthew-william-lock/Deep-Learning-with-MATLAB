function flipped_image = FlipImage(img)

    aa = 0:1:31;
    bb = 32:-1:1;
    vv = repmat(32*aa, 32, 1);
    ind_flip = vv(:) + repmat(bb', 32, 1);
    inds_flip = [ind_flip; 1024+ind_flip; 2048+ind_flip];
    flipped_image = img(inds_flip);
    
end

