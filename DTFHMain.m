function  DTFHMain

datasets = ["Oxford_5K","Paris_6K","Oxford_105K","Paris_106K","Holidays","UKB"];

layerList = ["pool5","pool5","out_relu","res5c_relu","efficientnet-b0|model|head|conv2d|Conv2D"];

net_name = ["vgg16","alexnet","mobilenetv2","resnet101","efficientnetb0"];

numList = [512,256,1280,2048,1280];

Index = 1;

dataname = datasets(Index);

layer = layerList(Index);

num = numList(Index);

net = eval(net_name(Index));

%%

filepatch = "G:/Datasets/" + dataname + "/";

filename = dir(filepatch + "*.jpg");

[file_num, ~] = size(filename);

%%
positiveCrow = zeros(file_num,num);

%%
parfor i=1:file_num
    
    im = imread("G:/Datasets/" + dataname + "/" + filename(i).name);
    
    if size(im,3)==1
        rgb = cat(3,im,im,im);
        im = mat2gray(rgb);
    end
    
    img = single(im);
    
    [h, w, ~] = size(img);
    
    if(h < 384 || w < 384)
        img_resize = imresize(img, [384 384]);
    else
        img_resize = img;
    end
    
%     if dataname=="Holidays"
%         img_resize = imresize(img, [786 786]);
%     end
    
    X =  activations(net, img_resize, layer, 'OutputAs',  'channels');
    
    positiveCrow(i,:) = GFCH_feature(X);
    
    %%
    i
    %%
end

positive_deepCrow = normalize(positiveCrow,2,'norm');


for i=1:file_num
    
    split = strsplit(filename(i).name, {'.'});
    
    name = split(1);
    
    %%
    positive_feature_save = positive_deepCrow(i,:)';
    
    save(['G:/Vectorsets/',dataname{1},'/deepTexture/',name{1},'.txt'],'positive_feature_save','-ASCII');
    
end

end
