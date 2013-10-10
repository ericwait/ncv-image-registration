
bigImSize = [512 512 120];

im1Lims = [bigImSize(1)/2-128 bigImSize(2)/2-128 bigImSize(3)/2-32; bigImSize(1)/2+128-1 bigImSize(2)/2+128-1 bigImSize(3)/2+32-1];
im2Lims = [bigImSize(1)/2-128 bigImSize(2)/2-128 bigImSize(3)/2-32; bigImSize(1)/2+128-1 bigImSize(2)/2+128-1 bigImSize(3)/2+32-1];

im1Lims = im1Lims - [60 50 1;60 50 1];
im2Lims = im2Lims + [10 10 2;10 10 2];

imdata = zeros(bigImSize);
imdata(round(bigImSize(1)/2-10):round(bigImSize(1)/2+10),:,round(bigImSize(3)/2-5):round(bigImSize(3)/2+5)) = 0.9;
imdata(:,round(bigImSize(2)/2-10):round(bigImSize(2)/2+10),round(bigImSize(3)/2-5):round(bigImSize(3)/2+5)) = 0.9;

h = normpdf(-4:4,0,3);
hx = h;
hy = h.';
hz = reshape(h,1,1,9);

smoothIm = convn(imdata,hx,'same');
smoothIm = convn(smoothIm,hy,'same');
smoothIm = convn(smoothIm,hz,'same');

im1 = smoothIm(im1Lims(1,1):im1Lims(2,1),im1Lims(1,2):im1Lims(2,2),im1Lims(1,3):im1Lims(2,3));
im2 = smoothIm(im2Lims(1,1):im2Lims(2,1),im2Lims(1,2):im2Lims(2,2),im2Lims(1,3):im2Lims(2,3));

% Perturb im1 with noise
im1 = im1 + normrnd(0,0.05,size(im1));
im1(im1 < 0) = 0;
im1(im1 > 1) = 1;

% Scale intensity
im1 = im1 * 0.8;

% Perturb im2 with noise
im2 = im2 + normrnd(0,0.05,size(im2));
im2(im2 < 0) = 0;
im2(im2 > 1) = 1;

figure;imagesc(max(smoothIm,[],3));colormap(gray);hold on;
rectangle('Position',[im1Lims(1,2) im1Lims(1,1) im1Lims(2,2)-im1Lims(1,2) im1Lims(2,1)-im1Lims(1,1)], 'EdgeColor','r');
rectangle('Position',[im2Lims(1,2) im2Lims(1,1) im2Lims(2,2)-im2Lims(1,2) im2Lims(2,1)-im2Lims(1,1)], 'EdgeColor','b');

% figure;imagesc(255*max(im1,[],3));colormap(gray);hold on;
% figure;imagesc(255*max(im2,[],3));colormap(gray);hold on;

mkdir('testRegistration1');
mkdir('testRegistration2');

%PerturbedPos
% delta1 = round(normrnd(0,30,[1,2]));
% delta2 = round(normrnd(0,30,[1,2]));
% 
% delta = delta2-delta1

delta1 = [2 3];
delta2 = [-4 -1];

fid = fopen(fullfile('testRegistration1', 'testRegistration1.txt'),'wt');
fprintf(fid, 'DatasetName:testRegistration1\n');
fprintf(fid, 'NumberOfChannels:1\n');
fprintf(fid, 'NumberOfFrames:1\n');
fprintf(fid, ['XDimension:' num2str(size(im1,2)) '\n']);
fprintf(fid, ['YDimension:' num2str(size(im1,1)) '\n']);
fprintf(fid, ['ZDimension:' num2str(size(im1,3)) '\n']);
fprintf(fid, 'XPixelPhysicalSize:1\n');
fprintf(fid, 'YPixelPhysicalSize:1\n');
fprintf(fid, 'ZPixelPhysicalSize:1\n');
% Reverse stage coordinates
% fprintf(fid, 'XPosition:%f\n', round(im1Lims(1,2)+delta1(1))/1e6);
% fprintf(fid, 'YPosition:%f\n', round(im1Lims(1,1)+delta1(2))/1e6);
fprintf(fid, 'XPosition:%f\n', round(im1Lims(1,1)+delta1(2))/1e6);
fprintf(fid, 'YPosition:%f\n', round(im1Lims(1,2)+delta1(1))/1e6);
fclose(fid);

fid = fopen(fullfile('testRegistration2', 'testRegistration2.txt'),'wt');
fprintf(fid, 'DatasetName:testRegistration2\n');
fprintf(fid, 'NumberOfChannels:1\n');
fprintf(fid, 'NumberOfFrames:1\n');
fprintf(fid, ['XDimension:' num2str(size(im2,2)) '\n']);
fprintf(fid, ['YDimension:' num2str(size(im2,1)) '\n']);
fprintf(fid, ['ZDimension:' num2str(size(im2,3)) '\n']);
fprintf(fid, 'XPixelPhysicalSize:1\n');
fprintf(fid, 'YPixelPhysicalSize:1\n');
fprintf(fid, 'ZPixelPhysicalSize:1\n');
% Reverse stage coordinates
% fprintf(fid, 'XPosition:%f\n', round(im2Lims(1,2)+delta2(1))/1e6);
% fprintf(fid, 'YPosition:%f\n', round(im2Lims(1,1)+delta2(2))/1e6);
fprintf(fid, 'XPosition:%f\n', round(im2Lims(1,1)+delta2(2))/1e6);
fprintf(fid, 'YPosition:%f\n', round(im2Lims(1,2)+delta2(1))/1e6);
fclose(fid);


for z=1:size(im1,3)
    imageNameStr = '_c%d_t%04d_z%04d.tif';
    
    imageNameSuffix = sprintf(imageNameStr, 1, 1, z);
    imwrite(im1(:,:,z), fullfile('testRegistration1',['testRegistration1' imageNameSuffix]), 'TIF');
    imwrite(im2(:,:,z), fullfile('testRegistration2',['testRegistration2' imageNameSuffix]), 'TIF');
end

fid = fopen('list.txt','wt');
fprintf(fid, 'testRegistration1\ntestRegistration2\n');
fclose(fid);
