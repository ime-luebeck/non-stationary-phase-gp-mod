% EIT_LOAD_DATA
% 
% Description
%   Load EIT dataset and reconstruct images.
%
%   Download eidors-v3.9.1 from http://eidors3d.sourceforge.net/download.shtml 
%   and extract all files into the ‘demos/demo_EIT/’ directory.
%
%   Download ready made FEM models from https://iweb.dl.sourceforge.net/project/eidors3d/eidors-v3/eidors-v3.6/model_library.zip
%   and extract the files into the directory ‘/eidors-v3.9.1/eidors/models/cache’.
%
%   Download data from http://eidors3d.sourceforge.net/data_contrib/if-neonate-spontaneous/index.shtml 
%   and copy the data into the ‘demos/demo_EIT’ directory
%
%   Run ‘eit_load_data.m’ to solve the EIT reconstruction. A mat file 
%   containing the EIT images is generated and stored into the 
%   ‘demos/demo_EIT/if-neonate-spontaneous’ directory.

% load EIT stuff
run ./demos/demo_EIT/eidors-v3.9.1/eidors/startup.m

% load EIT model
load('./demos/demo_EIT/eidors-v3.9.1/eidors/models/cache/neonate_boundary_EP_16_1_0.5_ES_0.1_maxsz_0.08.mat')
[fmdl.stimulation,fmdl.meas_select] = mk_stim_patterns(16,1,'{ad}','{ad}');
opt.imgsz = [64 64]; opt.noise_figure = 0.5;
imdl = mk_GREIT_model(mk_image(fmdl,1), 0.25, [], opt);

%% reconstruct images
vv= eidors_readdata('./demos/demo_EIT/if-neonate-spontaneous/P04P-1016.get'); vi=vv(:,45); vh=vv(:,61);
imr = inv_solve(imdl,vh,vi);

clf; axes('position',[0.05,0.5,0.25,0.45]);
imr.calc_colours = struct('ref_level',0,'greylev',0.2,'backgnd',[1,1,1]);
show_slices(imr);

yposns = [45  20 50]; xposns = [50  40 27]; ofs= [0,22,15];

% Show positions on image
hold on; for i = 1:length(xposns)
    plot(xposns(i),yposns(i),'s','LineWidth',10);
end; hold off;

% Show plots
imgs = calc_slices(inv_solve(imdl, vh, vv));
axes('position',[0.32,0.6,0.63,0.25]);

taxis =  (0:size(imgs,3)-1)/13; % frame rate = 13
hold all
for i = 1:length(xposns);
    plot(taxis,ofs(i)+squeeze(imgs(yposns(i),xposns(i),:)),'LineWidth',2);
end
hold off
set(gca,'yticklabel',[]); xlim([0 16]);

%print_convert neonate_intro04a.jpg

%% animate images
max_im = max(max(max(imgs)));
min_im = min(min(min(imgs)));
implay(mat2gray(imgs, [max_im, min_im]), 13);

%% store 
save('./demos/demo_EIT/if-neonate-spontaneous/eit_imgs.mat', 'imgs')

