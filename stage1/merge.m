clear all
clc
addpath('D:/caffe-windows/Build/x64/Release/matcaffe/');
    
phase = 'test';

    
% branch1 8x8
branch1_deploy = './branch1.prototxt';
branch1_weight = './branch1.caffemodel';
model(1) = caffe.Net(branch1_deploy,branch1_weight,phase);

% branch2 12x12
branch2_deploy = './branch2.prototxt';
branch2_weight = './branch2.caffemodel';
model(2) = caffe.Net(branch2_deploy,branch2_weight,phase);

% branch3 16x16
branch3_deploy = './branch3.prototxt';
branch3_weight = './branch3.caffemodel';
model(3) = caffe.Net(branch3_deploy,branch3_weight,phase);

% branch3 24x24
branch4_deploy = './branch4.prototxt';
branch4_weight = './branch4.caffemodel';
model(4) = caffe.Net(branch4_deploy,branch4_weight,phase);

% branch5 32x32
branch5_deploy = './branch5.prototxt';
branch5_weight = './branch5.caffemodel';
model(5) = caffe.Net(branch5_deploy,branch5_weight,phase);

% branch6 48x48
branch6_deploy = './branch6.prototxt';
branch6_weight = './branch6.caffemodel';
model(6) = caffe.Net(branch6_deploy,branch6_weight,phase);

% branch7 64x64
branch7_deploy = './branch7.prototxt';
branch7_weight = './branch7.caffemodel';
model(7) = caffe.Net(branch7_deploy,branch7_weight,phase);

% branch8 96x96
branch8_deploy = './branch8.prototxt';
branch8_weight = './branch8.caffemodel';
model(8) = caffe.Net(branch8_deploy,branch8_weight,phase);

% branch9 128x128
branch9_deploy = './branch9.prototxt';
branch9_weight = './branch9.caffemodel';
model(9) = caffe.Net(branch9_deploy,branch9_weight,phase);

% branch11 192x192
branch10_deploy = './branch10.prototxt';
branch10_weight = './branch10.caffemodel';
model(10) = caffe.Net(branch10_deploy,branch10_weight,phase);

% branch11 256x256
branch11_deploy = './branch11.prototxt';
branch11_weight = './branch11.caffemodel';
model(11) = caffe.Net(branch11_deploy,branch11_weight,phase);


merge_model = caffe.Net('stage1.prototxt',branch1_weight, phase);

for i=2:11
        layer_name = model(i).layer_names{2};
        merge_model.params(layer_name,1).set_data(model(i).params(layer_name,1).get_data());
        merge_model.params(layer_name,2).set_data(model(i).params(layer_name,2).get_data());
        layer_name = model(i).layer_names{3};
        merge_model.params(layer_name,1).set_data(model(i).params(layer_name,1).get_data());
        layer_name = model(i).layer_names{4};
        merge_model.params(layer_name,1).set_data(model(i).params(layer_name,1).get_data());
        merge_model.params(layer_name,2).set_data(model(i).params(layer_name,2).get_data());
end
merge_model.save('stage1.caffemodel');



