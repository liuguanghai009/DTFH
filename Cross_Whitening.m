function Cross_Whitening

Tests = ["Oxford_5K","Paris_6K","Oxford_105K","Paris_106K","Holidays","UKB"];
Trains = ["Paris_6K","Oxford_5K","Paris_6K","Oxford_5K","Paris_106K","UKB"];

Index = 1;

dataXTest = Tests(Index);
dataXTrain = Trains(Index);

num = 512;

%%
filesXTrain = dir("G:/Vectorsets/" + dataXTrain + "/deepTexture/" + "*.txt");
filename = dir("G:/Vectorsets/" + dataXTest + "/deepTexture/" + "*.txt");

[file_count, ~] = size(filesXTrain);
[file_num, ~] = size(filename);

XTrain = zeros(file_count,num);

XTest = zeros(file_num,num);

for i=1:file_count
    XTrain(i,:) = importdata("G:/Vectorsets/" + dataXTrain{1} + "/deepTexture/" + filesXTrain(i).name);
    %%
    i
    %%
end

for i=1:file_num
    XTest(i,:) = importdata("G:/Vectorsets/" + dataXTest{1} + "/deepTexture/" + filename(i).name);
    %%
    i
    %%
end

X = transfer_learning_whitening(XTrain,XTest,512);

% X = PCA(XTest,512);

%%

for i=1:file_num
    
    split = strsplit(filename(i).name, {'.'});
    
    name = split(1);
    %%
    feature_save = X(i,:)';
    save(['G:/Vectorsets/',dataXTest{1},'/DTFH/',name{1},'.txt'],'feature_save','-ASCII');
    %%
end

end