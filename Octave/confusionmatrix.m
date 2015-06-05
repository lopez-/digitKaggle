function confmat = confusionmatrix(y, predicted)

  difClasses = size(unique(y))(1);
  
  confmat = zeros(difClasses);
  
  for pred = 1:difClasses
    
    for c = 1:difClasses
      
      confmat(pred,c) = (sum(predicted(find(y==(c-1)),:)==(c-1)))/(sum(y==(c-1)));
      
    end;
    
  end;
  
end