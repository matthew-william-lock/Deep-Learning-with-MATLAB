%% 01 Read the data

book_fname = 'dataset/goblet_book.txt';
fid = fopen(book_fname,'r');
book_data = fscanf(fid,'%c');
fclose(fid);

%% Get unique charactars

% Get unique characters and input dimensionality
book_chars = unique(book_data);
K = length(book_chars);

% Mapping
char_to_ind = containers.Map('KeyType','char','ValueType','int32');
ind_to_char = containers.Map('KeyType','int32','ValueType','char');

% Create mappings
for i = 1:K
    char_to_ind(book_chars(i)) = i;
    ind_to_char(i) = book_chars(i);
end