function features_pca = stanford_PCA_whitening(features,dim)

x = features';

avg = mean(x, 1);

x = x - repmat(avg, size(x,1), 1);

%%

TF = isnan(x);

x(TF) = 0;

%%

sigma = (x*x')/size(x,2);

[U,~,~] = svd(sigma);

Xpca = U(:,1:dim)' * x;

%%

sigma = (Xpca * Xpca')/size(Xpca,2);

[u,s,~] = svd(sigma);

xRot = u'* Xpca;

epsilon = 1e-5;

xPCAWhite = diag(1./(sqrt(diag(s)+epsilon)))*xRot;

x2 = xPCAWhite';

%%

features_pca = normalize(x2,2,'norm');

end