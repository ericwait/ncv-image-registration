function [ ncv ] = NormalizedCovariance(im1, im2)
%NORMALIZEDCOVARIANCE Summary of this function goes here
%   Detailed explanation goes here

im1 = double(im1);
im2 = double(im2);
sig1 = sqrt(var(im1(:)));
sig2 = sqrt(var(im2(:)));

mean1 = mean(im1(:));
mean2 = mean(im2(:));

imSub1 = im1 - mean1;
imSub2 = im2 - mean2;

imMul = imSub1.*imSub2;

numerator = sum(imMul(:));

ncv = numerator / (numel(im1)*sig1*sig2);
end

