randomGeneration = randi(size(X)(1),4,1);

randomSample = X(randomGeneration,:);

for i = 1:4
  subplot(2,2,i), imagesc(reshape(randomSample(i,:),28,28));
end
colormap(gray);