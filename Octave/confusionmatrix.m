function [confmat] = confusionmatrix(actual, predicted)
  
  difClasses = size(unique(actual))(1);
  
  confmat = zeros(difClasses);
  
  for aClass = 1:difClasses
    
    for pClass = 1:difClasses
      
      confmat(aClass,pClass) = 100*(sum(predicted(find(actual==aClass),:)==pClass))/(sum(actual==aClass));
      
    end;
    
    confmat = round(confmat);
    
  end;
  
  imagesc(confmat);
  colormap(winter);
  
  for aClass = 1:difClasses
    
    for pClass = 1:difClasses
    
      strToDisplay = strcat(int2str(confmat(aClass,pClass)),'%');
      
      text((aClass-0.2),pClass, strToDisplay, 'FontName', 'verdana', 'FontSize', 10, 'Color', 'black');
    
    end;
    
  end;
  
  xlabel('Prediction');
  ylabel('Actual');
  
end