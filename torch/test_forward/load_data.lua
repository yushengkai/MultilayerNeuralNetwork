filename = '../../data/unittest.dat'
fid = io.open(filename, 'r')
line = ''
while true do
    line = fid:read('*l')
    if line == nil then
        break
    end
    for word in string.gmatch(line, '%d+') do
        print(word)
    end

end

fid:close()

