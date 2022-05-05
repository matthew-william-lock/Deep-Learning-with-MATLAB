function names = LoadLabelNames(filename)
    
    % Load batch
    A = load(filename);
    names =  A.label_names;

end

