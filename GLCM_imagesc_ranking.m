function GLCM_imagesc_ranking

net = vgg16;

im = imread("G:/HW.jpeg");

if size(im,3)==1
    rgb=cat(3,im,im,im);
    im=mat2gray(rgb);
end

img = single(im);

[h, w, ~] = size(img);

if(h < 384||w < 384)
    img_resize = imresize(img, [384 384]);
else
    img_resize=img;
end

pool5 = activations(net,img_resize,'pool5','OutputAs','channels');

[~,~,K] = size(pool5);

glcm = zeros(14,K);
rankw = zeros(1,K);

for i=1:K
    glcm(:,i)=  GLCM(pool5(:,:,i));
    rankw(1,i) = glcm(1,i);
end

[~,index] = sort(rankw,'descend');

figure;
subplot(4, 11, 1);

imagesc(im);
axis off;

for v = 1:10
    subplot(4, 11, v+1);
    S = imresize(pool5(:,:,index(v)), [h w]);
    imagesc(S);
    axis off;
end

for v = 11:20
    subplot(4, 11, v+2);
    S = imresize(pool5(:,:,index(v)), [h w]);
    imagesc(S);
    axis off;
end

subplot(4, 11, 23);
axis off;

for v = 21:30
    subplot(4, 11, v+3);
    S = imresize(pool5(:,:,index(492+(v-20))), [h w]);
    imagesc(S);
    axis off;
end

for v =31:40
    subplot(4, 11, v+4);
    S = imresize(pool5(:,:,index(492+(v-20))), [h w]);
    imagesc(S);
    axis off;
end

end
