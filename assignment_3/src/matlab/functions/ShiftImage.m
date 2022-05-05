function xx_shifted = ShiftImage(image, tx, ty)

    aa = 0:1:31;
    vv = repmat(32*aa, 32-tx, 1);

    bb1 = tx+1:1:32;
    bb2 = 1:32-tx;
    
    ind_fill = vv(:) + repmat(bb1', 32, 1);
    ind_xx = vv(:) + repmat(bb2', 32, 1);

    ii = find(ind_fill >= ty*32+1);
    ind_fill = ind_fill(ii(1):end);
    
    ii = find(ind_xx <= 1024-ty*32);
    ind_xx = ind_xx(1:ii(end));
    inds_fill = [ind_fill; 1024+ind_fill; 2048+ind_fill];
    inds_xx = [ind_xx; 1024+ind_xx; 2048+ind_xx];

    xx_shifted(inds_fill) = image(inds_xx);




    
    
    
    
    



    % show shifted image
    % image = permute(reshape(xx_shifted',32, 32, 3),[2,1,3])


end