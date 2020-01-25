function store_video(file, imgs)

outputVideo = VideoWriter(fullfile(file));
outputVideo.FrameRate = 13;
open(outputVideo)
max_val = max(imgs(:));
min_val = min(imgs(:));
img_gray = mat2gray(imgs, [min_val, max_val]);

max_val_gray = max(img_gray(:));
min_val_gray = min(img_gray(:));

f = figure();
for i = 1:size(imgs,3)
    I = kron(img_gray(:,:,i),ones(10));
   I(isnan(I)) = max_val;

%    imagesc(I, [min_val_gray, max_val_gray])
%    truesize(f)
%    %colormap(flip(parula))
%    colormap(flip(gray))
%    axis image

    imshow(max_val-kron(imgs(:,:,i),ones(10)), [0, max_val-min_val])
   
   frame = getframe;
   writeVideo(outputVideo,frame);
end

close(outputVideo)

end