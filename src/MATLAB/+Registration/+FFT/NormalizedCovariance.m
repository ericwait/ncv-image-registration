% ***********************************************************************
%      Copyright 2011-2016 Drexel University

%      Registration is free software: you can redistribute it and/or modify
%      it under the terms of the GNU General Public License as published by
%      the Free Software Foundation, either version 3 of the License, or
%      (at your option) any later version.

%      Registration is distributed in the hope that it will be useful,
%      but WITHOUT ANY WARRANTY; without even the implied warranty of
%      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%      GNU General Public License for more details.

%      You should have received a copy of the GNU General Public License
%      along with Registration in file "gnu gpl v3.txt".  If not, see
%      <http://www.gnu.org/licenses/>.

% ***********************************************************************/

function [ NCshiftMatrix ] = NormalizedCovariance( im1,im2, minOverlap, im1Mask, im2Mask )
%NORMALIZEDCOVARIANCE Summary of this function goes here
%   Detailed explanation goes here

%% Sanitize Inputs.
msgID = 'getMeasurementCC:NotSameDimensions';
msg = 'This function can take only up to 2 inputs.';
notSameDimensionsException = MException( msgID, msg );
if nargin == 2
    remsize = 10;
end
sizeIm1 = size( im1 );
sizeIm2 = size( im2 );
dimIm1 = ndims( im1 );
dimIm2 = ndims( im2 );

if dimIm1 ~= dimIm2
    throw(notSameDimensionsException);
end

if (~exist('im1Mask','var') || isempty(im1Mask))
    im1Mask = ones(sizeIm1);
else
    im1Mask = im2double(im1Mask);
end
if (~exist('im2Mask','var') || isempty(im2Mask))
    im2Mask = ones(sizeIm2);
else
    im2Mask = im2double(im2Mask);
end

im1(~im1Mask) = 0;
im2(~im2Mask) = 0;

%% Set up Fourier Transform.
EPSILON = 1e-7;

sizeBigImage = sizeIm1 + sizeIm2;
im2flip = im2;
im2MaskFlip = im2Mask;
% Flip second image to get correlation instead of convolution.
for d = 1:dimIm2
    im2flip = flip(im2flip,d);
    im2MaskFlip = flip(im2MaskFlip,d);
end

% Pad by zeros to make circular convolation equivalent to linear.
imEmb1 = zeros( sizeBigImage );
imEmb2 = zeros( sizeBigImage );
imEmb1ones = zeros( sizeBigImage );
imEmb2ones = zeros( sizeBigImage );
subsCell1 = cell(1, dimIm1 );
subsCell2 = cell(1, dimIm2 );
for d = 1:dimIm1
    subsCell1{d} = 1:sizeIm1(d);
    subsCell2{d} = 1:sizeIm2(d);
end

imEmb1(subsCell1{:}) = im1;
imEmb2(subsCell2{:}) = im2flip;
imEmb1ones(subsCell1{:}) = im1Mask;
imEmb2ones(subsCell2{:}) = im2MaskFlip;

%% Compute the Fourier transforms
% Transform images.
im1fft = fftn(imEmb1);
im2fft = fftn(imEmb2);

% Transform all one images to compute sums.
im1fftOnes = fftn(imEmb1ones);
im2fftOnes = fftn(imEmb2ones);

clear imEmb1ones;
clear imEmb2ones;

% Transform squared images to get variances.
im1fftSq = fftn(imEmb1.^2);
im2fftSq = fftn(imEmb2.^2);

clear imEmb1;
clear imEmb2;

% Transform Overlap Sumproducts for covariances.
SumOverlapIm = ifftn( im1fft.*im2fft );
% SumOverlapIm(SumOverlapIm<0) = 0;

% Compute sizes of the overlaps.
Noverlap = round(ifftn( im1fftOnes.*im2fftOnes ));

% Compute Sum Products of overlaps.
sumProd1 = ifftn( im1fft.*im2fftOnes );
sumProd2 = ifftn( im1fftOnes.*im2fft );

clear im1fft;
clear im2fft;

% Compute Overlap variance number of pixels.
varIm1 = Noverlap.*ifftn( im1fftSq.*im2fftOnes ) - sumProd1.^2;
varIm2 = Noverlap.*ifftn( im2fftSq.*im1fftOnes ) - sumProd2.^2;

clear im1fftOnes;
clear im2fftOnes;
clear im1fftSq;
clear im2fftSq;

% Compute Overlap Covariace and geometric mean of variances.
covIm1Im2 = Noverlap.*SumOverlapIm - sumProd1.*sumProd2;
sqrtVar1Var2 = sqrt(varIm1.*(varIm1>EPSILON)).*sqrt(varIm2.*(varIm2>EPSILON));

clear SumOverlapIm;
clear sumProd1;
clear sumProd2;
clear varIm1;
clear varIm2;

% Compute Normalized Covariance
NormCov = zeros(size(covIm1Im2));
NormCov(sqrtVar1Var2>0) = covIm1Im2(sqrtVar1Var2>0)./...
    sqrtVar1Var2(sqrtVar1Var2>0);

clear covIm1Im2;
clear sqrtVar1Var2;

% Remove noisy peaks.
NormCov(abs(NormCov)>1) = 0;
NormCov( Noverlap < minOverlap )=0;

% Output.
NCshiftMatrix = NormCov;
end

