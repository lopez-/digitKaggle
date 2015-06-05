randomGeneration = randi(size(X)(1),4,1);

randomSample = X(randomGeneration,:);

for i = 1:4
  subplot(2,2,i), imagesc(reshape(randomSample(i,:),28,28));
  text(1,25,num2str(i),'FontSize',30,'Color','white');
end
colormap(gray);