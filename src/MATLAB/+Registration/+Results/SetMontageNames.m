faceColor2mo = [0,1,0];
marker2mo = 'v';
faceColor22mo = [1,0,0];
marker22mo = 's';
faceColorCtrl = [0,0,1];
markerCtrl = 'o';

montages = struct('filePath','','chanList',{[]},'faceColor',{[0,0,0]},'marker','');

rootDir = 'P:\Images\Temple\3d\SVZ\Montage\PumpData\';


    subDir = 'Contralateral Sides 7-2-15';

        curDir = fullfile('22mSVZ w2mChP-CM',...
            '22mSVZ w2mChP-CM #_Montage_wDelta\');
            montages(end).filePath = fullfile(rootDir,subDir,curDir);
            montages(end).chanList = [1,2,4];
            montages(end).faceColor = faceColor2mo;
            montages(end).marker = marker2mo;
            
        curDir = fullfile('22mSVZ w22mChP-CM',...
            '22mSVZ w22mChP-CM #_Montage_wDelta\');
            montages(end+1).filePath = fullfile(rootDir,subDir,curDir);
            montages(end).chanList = [1,2,4];
            montages(end).faceColor = faceColor22mo;
            montages(end).marker = marker22mo;
       
        curDir = fullfile('22mSVZ wControlPump1 10x1',...
            '22mSVZ wControlPump1 10x1 #_Montage_wDelta\');
            montages(end+1).filePath = fullfile(rootDir,subDir,curDir);
            montages(end).chanList = [1,2,4];
            montages(end).faceColor = faceColorCtrl;
            montages(end).marker = markerCtrl;
            
        curDir = fullfile('22mSVZ wControlPump1 10x2',...
            '22mSVZ wControlPump1 10x2 #_Montage_wDelta\');
            montages(end+1).filePath = fullfile(rootDir,subDir,curDir);
            montages(end).chanList = [1,2,4];
            montages(end).faceColor = faceColorCtrl;
            montages(end).marker = markerCtrl;

            
    subDir = 'Contralateral Sides 9-18-15';
    
        curDir = fullfile('22mSVZ StabSham1 10x1_4x01',...
            '22mSVZ StabSham 10x1_4x01 #_Montage_wDelta\');
            montages(end+1).filePath = fullfile(rootDir,subDir,curDir);
            montages(end).chanList = [1,2,4];
            montages(end).faceColor = faceColorCtrl;
            montages(end).marker = markerCtrl;
            
        curDir = fullfile('22mSVZ w2mChP-CM 2 10x1_4x01',...
            '22mSVZ w2mChP-CM 2 10x1_4x01 #_Montage_wDelta\');
            montages(end+1).filePath = fullfile(rootDir,subDir,curDir);
            montages(end).chanList = [1,2,4];
            montages(end).faceColor = faceColor2mo;
            montages(end).marker = marker2mo;
            
        curDir = fullfile('22mSVZ w22mChP-CM 1 10x1_4x01',...
            '22mSVZ w22mChP-CM 1 10x1_4x01 #_Montage_wDelta\');
            montages(end+1).filePath = fullfile(rootDir,subDir,curDir);
            montages(end).chanList = [1,2,4];
            montages(end).faceColor = faceColor22mo;
            montages(end).marker = marker22mo;
            
        curDir = fullfile('22mSVZ w22mChP-CM 2 10x1_4x01',...
            '22mSVZ w22mChP-CM 2 10x1_4x01 #_Montage_wDelta\');
            montages(end+1).filePath = fullfile(rootDir,subDir,curDir);
            montages(end).chanList = [1,2,4];
            montages(end).faceColor = faceColor22mo;
            montages(end).marker = marker22mo;
  
            
    subDir = 'Contralateral wmSVZs 11-4-15';
    
        curDir = fullfile('22mSVZ 22mChP-CM1 10x1_4x01',...
            '22mSVZ 22mChP-CM1 10x1_4x01 #_Montage_wDelta\');
            montages(end+1).filePath = fullfile(rootDir,subDir,curDir);
            montages(end).chanList = [1,2,4];
            montages(end).faceColor = faceColor22mo;
            montages(end).marker = marker22mo;
            
        curDir = fullfile('22mSVZ 22mChP-CM2 10x1_4x01',...
            '22mSVZ 22mChP-CM2 10x1_4x01 #_Montage_wDelta\');
            montages(end+1).filePath = fullfile(rootDir,subDir,curDir);
            montages(end).chanList = [1,2,4];
            montages(end).faceColor = faceColor22mo;
            montages(end).marker = marker22mo;
            
        curDir = fullfile('22mSVZ StabSham1 10x1_4x01',...
            '22mSVZ StabSham1 10x1_4x01 #_Montage_wDelta\');
            montages(end+1).filePath = fullfile(rootDir,subDir,curDir);
            montages(end).chanList = [1,2,4];
            montages(end).faceColor = faceColorCtrl;
            montages(end).marker = markerCtrl;
            
        curDir = fullfile('22mSVZ StabSham1 10x2_4x01',...
            '22mSVZ StabSham1 10x2_4x01 #_Montage_wDelta\');
            montages(end+1).filePath = fullfile(rootDir,subDir,curDir);
            montages(end).chanList = [1,2,4];
            montages(end).faceColor = faceColorCtrl;
            montages(end).marker = markerCtrl;
            
            
    subDir = 'DAPI EdU-647 GFAP-488 Dcx-Cy3 12-22-2015';
        curDir = fullfile('22mSVZ StabSham Contra 12-26-15 OptSettings',...
            '22mSVZ StabSham Contra 12-26-15 OptSettings #_Montage_wDelta\');
            montages(end+1).filePath = fullfile(rootDir,subDir,curDir);
            montages(end).chanList = [1,2,4];
            montages(end).faceColor = faceColorCtrl;
            montages(end).marker = markerCtrl;
            
        curDir = fullfile('22mSVZ w2mChP-CM Y1 Contra 12-24-15 OptSettings',...
            '22mSVZ w2mChP-CM Y1 Contra 12-24-15 OptSettings #_Montage_wDelta\');
            montages(end).filePath = fullfile(rootDir,subDir,curDir);
            montages(end).chanList = [1,2,4];
            montages(end).faceColor = faceColor2mo;
            montages(end).marker = marker2mo;
            
        curDir = fullfile('22mSVZ w2mChP-CM Y3 Contra 12-22-15 OptSettings',...
            '22mSVZ w2mChP-CM Y3 Contra 12-22-15 OptSettings #_Montage_wDelta\');
            montages(end).filePath = fullfile(rootDir,subDir,curDir);
            montages(end).chanList = [1,2,4];
            montages(end).faceColor = faceColor2mo;
            montages(end).marker = marker2mo;
            
            
save('montages.mat','montages');
