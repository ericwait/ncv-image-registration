function [ NCshiftMatrix ] = NormalizedCovariance( im1,im2, minOverlap )
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

%% Set up Fourier Transform.
sizeBigImage = sizeIm1 + sizeIm2;
im2flip = im2;
% Flip second image to get correlation instead of convolution.
for d = 1:dimIm2
    im2flip = flip(im2flip,d);
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
refStruct1 = struct('type','()','subs',{subsCell1});
refStruct2 = struct('type','()','subs',{subsCell2});
imEmb1 = subsasgn( imEmb1, refStruct1, im1 );
imEmb2 = subsasgn( imEmb2, refStruct2, im2flip );
imEmb1ones = subsasgn( imEmb1ones, refStruct1, ones(sizeIm1) );
imEmb2ones = subsasgn( imEmb2ones, refStruct2, ones(sizeIm2) );

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
SumOverlapIm(SumOverlapIm<0) = 0;

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
sqrtVar1Var2 = sqrt(varIm1.*(varIm1>0)).*sqrt(varIm2.*(varIm2>0));

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
NormCov(NormCov>1) = 0;
NormCov( Noverlap < minOverlap )=0;

% Output.
NCshiftMatrix = NormCov;
end

