function [x] = GLCM(I)

glcm = graycomatrix(I, 'offset', [0 1], 'Symmetric', true);

x = haralickTextureFeatures(glcm);

end