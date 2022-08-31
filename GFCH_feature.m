function GFCH = GFCH_feature(X)

[hei,wid,K] = size(X);

glcm = zeros(14,K);
rankw = zeros(1,K);

for i=1:K
    glcm(:,i)=  GLCM(X(:,:,i));
    rankw(1,i) = glcm(1,i);
end

[~,index] = sort(rankw,'descend');

oNum = 6;
gabor_energy = zeros(hei,wid,oNum);

for v = 1:K/2
    for n=1:oNum
        ori = pi*(n-1)/oNum;
        gabor_energy(:,:,n) = gaborEnergy(X(:,:,index(v)),ori,2.3333);
    end
    energymap = max(gabor_energy,[],3).^2 + min(gabor_energy,[],3).^2;
    X(:,:,index(v))= sqrt(energymap).*rankw(1,index(v));
end

for v = K/2+1:K
    energymap = zeros(hei,wid);
    for n=1:oNum
        ori = pi*(n-1)/oNum;
        gabor_energy(:,:,n) = gaborEnergy(X(:,:,index(v)),ori,2.3333);
        energymap = energymap + gabor_energy(:,:,n).^2;
    end
    X(:,:,index(v)) = sqrt(energymap).*rankw(1,index(v));
end

GFCH = sum(X,[1 2]);

end