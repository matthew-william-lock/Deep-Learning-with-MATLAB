function P = EvaluateClassifier(X, W, b)

    intermediate = W*X + b;
    P = exp(intermediate)./sum(exp(intermediate));

end