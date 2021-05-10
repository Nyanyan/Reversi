openingsDat = io.open('openings.dat','r')
outFile = io.open('openings.csv','w')

for i = 1,396 do 
	line = openingsDat:read()
	for j = 1,#line,2 do
		local char = line:sub(j,j)
		local num  = line:sub(j+1,j+1)
		local col 
		local row
		if char == 'A' or char == 'a' then col = 1 end
		if char == 'B' or char == 'b' then col = 2 end
		if char == 'C' or char == 'c' then col = 3 end
		if char == 'D' or char == 'd' then col = 4 end
		if char == 'E' or char == 'e' then col = 5 end
		if char == 'F' or char == 'f' then col = 6 end
		if char == 'G' or char == 'g' then col = 7 end
		if char == 'H' or char == 'h' then col = 8 end
		row = tonumber(num)
		local ind = (row -1)*8 + col - 1
		outFile:write(tostring(ind))
		if line:sub(j+2,j+2) == ' ' then
			break
		else
			outFile:write(',')
		end
	end
	outFile:write('\n')
end

outFile:close()
openingsDat:close()
