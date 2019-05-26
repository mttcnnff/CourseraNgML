function [encoding] = binVec(num_labels,value)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

res = zeros(1, num_labels);
res(value) = 1;
encoding = res;
end

