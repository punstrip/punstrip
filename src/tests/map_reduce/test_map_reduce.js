var symbols = require('./test_symbols.js');
var map = require('./map.js');
var reduce = require('./reduce.js');

//symbol['sum'] = 1;
for(var i = 0; i < symbols.length; ++i){
	symbols[i]['sum'] = 1;
}

var best = reduce(1, symbols);
console.log(best);
