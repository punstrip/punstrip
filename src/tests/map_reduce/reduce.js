module.exports = function(key, values){
            
    var similarity = function(b){
        var similarities = [];

        var abs_diff = function( A, B, s){
            var size_max = Math.max( A, B );
            var abs_diff = Math.sqrt( Math.pow( (A - B), 2 ) )
            return ( Math.pow( size_max , s ) - Math.pow( abs_diff, s) )/ Math.pow( size_max, s );
        }

        var assert_score = function( score ){
            if( typeof(score) !== typeof(1.0)){
                throw 'Error, mismatch of score type :: ' + score;
            }
            if( score > 1.0  || score < 0.0){
                throw 'Error, score is out of bounds :: ' + score;
            }
        }

	Set.prototype.isSuperset = function(subset) {
	    for (var elem of subset) {
		if (!this.has(elem)) {
		    return false;
		}
	    }
	    return true;
	}
	Set.prototype.union = function(setB) {
	    var union = new Set(this);
	    for (var elem of setB) {
		union.add(elem);
	    }
	    return union;
	}
	Set.prototype.intersection = function(setB) {
	    var intersection = new Set();
	    for (var elem of setB) {
		if (this.has(elem)) {
		    intersection.add(elem);
		}
	    }
	    return intersection;
	}
	Set.prototype.difference = function(setB) {
	    var difference = new Set(this);
	    for (var elem of setB) {
		difference.delete(elem);
	    }
	    return difference;
	}


        //SIZE SIM
        if( 44 > 0 ){
            var size_score = abs_diff( 44, b.size, 0.8);
            assert_score( size_score );
            similarities.push( size_score );
        }else {
            similarities.push( 0.0 );
        }

        //HASH SIM
        var hash_sim = (( "a018715bd2029ceb0c03bf9973b8a063c98711898858653810019b268d52630d" == b.hash) ? 1.0 : 0.0 );
        assert_score( hash_sim );
        similarities.push( hash_sim );

        //OPCODE HASH SIM
        var opcode_hash_sim = (( "99d2482ee00c6278f1c2f5848aa7604a18868cb084a21423559718d56b1cc007" == b.opcode_hash) ? 1.0 : 0.0 );
        assert_score( opcode_hash_sim );
        similarities.push( opcode_hash_sim );

        var vex = {"temp_var_types": ["Ity_I8", "Ity_I8", "Ity_I8", "Ity_I64", "Ity_I8", "Ity_I64", "Ity_I64", "Ity_I64", "Ity_I64", "Ity_I32", "Ity_I32", "Ity_I32", "Ity_I32", "Ity_I32", "Ity_I32", "Ity_I64", "Ity_I64", "Ity_I8", "Ity_I1", "Ity_I64", "Ity_I64", "Ity_I64", "Ity_I64", "Ity_I64", "Ity_I64", "Ity_I32", "Ity_I8", "Ity_I64", "Ity_I32", "Ity_I8", "Ity_I32", "Ity_I64", "Ity_I64", "Ity_I64", "Ity_I64", "Ity_I32", "Ity_I64", "Ity_I64", "Ity_I64", "Ity_I1", "Ity_I64", "Ity_I64", "Ity_I64", "Ity_I64", "Ity_I64", "Ity_I64", "Ity_I64", "Ity_I64", "Ity_I1", "Ity_I8", "Ity_I8", "Ity_I1", "Ity_I8", "Ity_I32", "Ity_I64", "Ity_I32", "Ity_I64", "Ity_I32", "Ity_I64", "Ity_I32", "Ity_I64", "Ity_I64", "Ity_I1", "Ity_I32", "Ity_I32", "Ity_I1"], "ntemp_vars": 66, "statements": [{"tag": "Ist_IMark", "value": "------ IMark(0x40189f, 3, 0) ------"}, {"tag": "Ist_WrTmp", "value": "t3 = GET:I64(offset=72)"}, {"tag": "Ist_WrTmp", "value": "t2 = LDle:I8(t3)"}, {"tag": "Ist_WrTmp", "value": "t46 = 8Uto64(t2)"}, {"tag": "Ist_WrTmp", "value": "t15 = t46"}, {"tag": "Ist_IMark", "value": "------ IMark(0x4018a2, 3, 0) ------"}, {"tag": "Ist_WrTmp", "value": "t49 = 64to8(0x000000000000002d)"}, {"tag": "Ist_WrTmp", "value": "t50 = 64to8(t15)"}, {"tag": "Ist_WrTmp", "value": "t48 = CmpEQ8(t50,t49)"}, {"tag": "Ist_WrTmp", "value": "t47 = 1Uto64(t48)"}, {"tag": "Ist_WrTmp", "value": "t23 = t47"}, {"tag": "Ist_WrTmp", "value": "t51 = 64to1(t23)"}, {"tag": "Ist_WrTmp", "value": "t18 = t51"}, {"tag": "Ist_WrTmp", "value": "t52 = 1Uto8(t18)"}, {"tag": "Ist_WrTmp", "value": "t17 = t52"}, {"tag": "Ist_Put", "value": "PUT(offset=16) = t17"}, {"tag": "Ist_IMark", "value": "------ IMark(0x4018a5, 3, 0) ------"}, {"tag": "Ist_WrTmp", "value": "t53 = 8Uto32(t17)"}, {"tag": "Ist_WrTmp", "value": "t25 = t53"}, {"tag": "Ist_WrTmp", "value": "t54 = 32Uto64(t25)"}, {"tag": "Ist_WrTmp", "value": "t24 = t54"}, {"tag": "Ist_IMark", "value": "------ IMark(0x4018a8, 3, 0) ------"}, {"tag": "Ist_WrTmp", "value": "t5 = Add64(t3,t24)"}, {"tag": "Ist_Put", "value": "PUT(offset=72) = t5"}, {"tag": "Ist_Put", "value": "PUT(offset=184) = 0x00000000004018ab"}, {"tag": "Ist_IMark", "value": "------ IMark(0x4018ab, 3, 0) ------"}, {"tag": "Ist_WrTmp", "value": "t29 = LDle:I8(t5)"}, {"tag": "Ist_WrTmp", "value": "t55 = 8Sto32(t29)"}, {"tag": "Ist_WrTmp", "value": "t28 = t55"}, {"tag": "Ist_WrTmp", "value": "t56 = 32Uto64(t28)"}, {"tag": "Ist_WrTmp", "value": "t27 = t56"}, {"tag": "Ist_IMark", "value": "------ IMark(0x4018ae, 3, 0) ------"}, {"tag": "Ist_WrTmp", "value": "t57 = 64to32(t27)"}, {"tag": "Ist_WrTmp", "value": "t30 = t57"}, {"tag": "Ist_WrTmp", "value": "t9 = Sub32(t30,0x00000030)"}, {"tag": "Ist_WrTmp", "value": "t58 = 32Uto64(t9)"}, {"tag": "Ist_WrTmp", "value": "t34 = t58"}, {"tag": "Ist_Put", "value": "PUT(offset=16) = t34"}, {"tag": "Ist_IMark", "value": "------ IMark(0x4018b1, 3, 0) ------"}, {"tag": "Ist_WrTmp", "value": "t59 = 64to32(t34)"}, {"tag": "Ist_WrTmp", "value": "t35 = t59"}, {"tag": "Ist_Put", "value": "PUT(offset=144) = 0x0000000000000007"}, {"tag": "Ist_WrTmp", "value": "t60 = 32Uto64(t35)"}, {"tag": "Ist_WrTmp", "value": "t37 = t60"}, {"tag": "Ist_Put", "value": "PUT(offset=152) = t37"}, {"tag": "Ist_Put", "value": "PUT(offset=160) = 0x0000000000000009"}, {"tag": "Ist_Put", "value": "PUT(offset=184) = 0x00000000004018b4"}, {"tag": "Ist_IMark", "value": "------ IMark(0x4018b4, 2, 0) ------"}, {"tag": "Ist_WrTmp", "value": "t63 = 64to32(0x0000000000000009)"}, {"tag": "Ist_WrTmp", "value": "t64 = 64to32(t37)"}, {"tag": "Ist_WrTmp", "value": "t62 = CmpLE32U(t64,t63)"}, {"tag": "Ist_WrTmp", "value": "t61 = 1Uto64(t62)"}, {"tag": "Ist_WrTmp", "value": "t44 = t61"}, {"tag": "Ist_WrTmp", "value": "t65 = 64to1(t44)"}, {"tag": "Ist_WrTmp", "value": "t39 = t65"}, {"tag": "Ist_Exit", "value": "if (t39) { PUT(offset=184) = 0x4018b6; Ijk_Boring }"}], "expressions": [{"tag": "Iex_Get", "value": "GET:I64(offset=72)"}, {"tag": "Iex_Load", "value": "LDle:I8(t3)"}, {"tag": "Iex_RdTmp", "value": "t3"}, {"tag": "Iex_Unop", "value": "8Uto64(t2)"}, {"tag": "Iex_RdTmp", "value": "t2"}, {"tag": "Iex_RdTmp", "value": "t46"}, {"tag": "Iex_Unop", "value": "64to8(0x000000000000002d)"}, {"tag": "Iex_Const", "value": "0x000000000000002d"}, {"tag": "Iex_Unop", "value": "64to8(t15)"}, {"tag": "Iex_RdTmp", "value": "t15"}, {"tag": "Iex_Binop", "value": "CmpEQ8(t50,t49)"}, {"tag": "Iex_RdTmp", "value": "t50"}, {"tag": "Iex_RdTmp", "value": "t49"}, {"tag": "Iex_Unop", "value": "1Uto64(t48)"}, {"tag": "Iex_RdTmp", "value": "t48"}, {"tag": "Iex_RdTmp", "value": "t47"}, {"tag": "Iex_Unop", "value": "64to1(t23)"}, {"tag": "Iex_RdTmp", "value": "t23"}, {"tag": "Iex_RdTmp", "value": "t51"}, {"tag": "Iex_Unop", "value": "1Uto8(t18)"}, {"tag": "Iex_RdTmp", "value": "t18"}, {"tag": "Iex_RdTmp", "value": "t52"}, {"tag": "Iex_RdTmp", "value": "t17"}, {"tag": "Iex_Unop", "value": "8Uto32(t17)"}, {"tag": "Iex_RdTmp", "value": "t17"}, {"tag": "Iex_RdTmp", "value": "t53"}, {"tag": "Iex_Unop", "value": "32Uto64(t25)"}, {"tag": "Iex_RdTmp", "value": "t25"}, {"tag": "Iex_RdTmp", "value": "t54"}, {"tag": "Iex_Binop", "value": "Add64(t3,t24)"}, {"tag": "Iex_RdTmp", "value": "t3"}, {"tag": "Iex_RdTmp", "value": "t24"}, {"tag": "Iex_RdTmp", "value": "t5"}, {"tag": "Iex_Const", "value": "0x00000000004018ab"}, {"tag": "Iex_Load", "value": "LDle:I8(t5)"}, {"tag": "Iex_RdTmp", "value": "t5"}, {"tag": "Iex_Unop", "value": "8Sto32(t29)"}, {"tag": "Iex_RdTmp", "value": "t29"}, {"tag": "Iex_RdTmp", "value": "t55"}, {"tag": "Iex_Unop", "value": "32Uto64(t28)"}, {"tag": "Iex_RdTmp", "value": "t28"}, {"tag": "Iex_RdTmp", "value": "t56"}, {"tag": "Iex_Unop", "value": "64to32(t27)"}, {"tag": "Iex_RdTmp", "value": "t27"}, {"tag": "Iex_RdTmp", "value": "t57"}, {"tag": "Iex_Binop", "value": "Sub32(t30,0x00000030)"}, {"tag": "Iex_RdTmp", "value": "t30"}, {"tag": "Iex_Const", "value": "0x00000030"}, {"tag": "Iex_Unop", "value": "32Uto64(t9)"}, {"tag": "Iex_RdTmp", "value": "t9"}, {"tag": "Iex_RdTmp", "value": "t58"}, {"tag": "Iex_RdTmp", "value": "t34"}, {"tag": "Iex_Unop", "value": "64to32(t34)"}, {"tag": "Iex_RdTmp", "value": "t34"}, {"tag": "Iex_RdTmp", "value": "t59"}, {"tag": "Iex_Const", "value": "0x0000000000000007"}, {"tag": "Iex_Unop", "value": "32Uto64(t35)"}, {"tag": "Iex_RdTmp", "value": "t35"}, {"tag": "Iex_RdTmp", "value": "t60"}, {"tag": "Iex_RdTmp", "value": "t37"}, {"tag": "Iex_Const", "value": "0x0000000000000009"}, {"tag": "Iex_Const", "value": "0x00000000004018b4"}, {"tag": "Iex_Unop", "value": "64to32(0x0000000000000009)"}, {"tag": "Iex_Const", "value": "0x0000000000000009"}, {"tag": "Iex_Unop", "value": "64to32(t37)"}, {"tag": "Iex_RdTmp", "value": "t37"}, {"tag": "Iex_Binop", "value": "CmpLE32U(t64,t63)"}, {"tag": "Iex_RdTmp", "value": "t64"}, {"tag": "Iex_RdTmp", "value": "t63"}, {"tag": "Iex_Unop", "value": "1Uto64(t62)"}, {"tag": "Iex_RdTmp", "value": "t62"}, {"tag": "Iex_RdTmp", "value": "t61"}, {"tag": "Iex_Unop", "value": "64to1(t44)"}, {"tag": "Iex_RdTmp", "value": "t44"}, {"tag": "Iex_RdTmp", "value": "t65"}, {"tag": "Iex_RdTmp", "value": "t39"}, {"tag": "Iex_Const", "value": "0x00000000004018c5"}], "jumpkind": "Ijk_Boring", "sum_operations": {"Iop_64to32": 4, "Iop_64to8": 2, "Iop_CmpEQ8": 1, "Iop_1Uto64": 2, "Iop_Sub32": 1, "Iop_CmpLE32U": 1, "Iop_Add64": 1, "Iop_64to1": 2, "Iop_32Uto64": 4, "Iop_1Uto8": 1, "Iop_8Sto32": 1, "Iop_8Uto32": 1, "Iop_8Uto64": 1}, "operations": ["Iop_8Uto64", "Iop_64to8", "Iop_64to8", "Iop_CmpEQ8", "Iop_1Uto64", "Iop_64to1", "Iop_1Uto8", "Iop_8Uto32", "Iop_32Uto64", "Iop_Add64", "Iop_8Sto32", "Iop_32Uto64", "Iop_64to32", "Iop_Sub32", "Iop_32Uto64", "Iop_64to32", "Iop_32Uto64", "Iop_64to32", "Iop_64to32", "Iop_CmpLE32U", "Iop_1Uto64", "Iop_64to1"], "sum_statements": {"Ist_Put": 8, "Ist_IMark": 8, "Ist_WrTmp": 39, "Ist_Exit": 1}, "constants": [{"value": "0x000000000000002d", "type": "Ity_I64"}, {"value": "0x00000000004018ab", "type": "Ity_I64"}, {"value": "0x00000030", "type": "Ity_I32"}, {"value": "0x0000000000000007", "type": "Ity_I64"}, {"value": "0x0000000000000009", "type": "Ity_I64"}, {"value": "0x00000000004018b4", "type": "Ity_I64"}, {"value": "0x0000000000000009", "type": "Ity_I64"}, {"value": "0x00000000004018c5", "type": "Ity_I64"}], "ninstructions": 8, "sum_expressions": {"Iex_Get": 1, "Iex_RdTmp": 44, "Iex_Load": 2, "Iex_Binop": 4, "Iex_Const": 8, "Iex_Unop": 18}};
        if( vex != null && b.vex != null ){

            //number of instruction
            var vex_size_score = abs_diff( vex.ninstructions, b.vex.ninstructions, 1.7);
            assert_score( vex_size_score );
            similarities.push( vex_size_score );

            //number of temporary variables
            var vex_ntemp_score = abs_diff( vex.ntemp_vars, b.vex.ntemp_vars, 1.1);
            assert_score( vex_ntemp_score );
            similarities.push( vex_ntemp_score );

            //jump kind
            var vex_jumpkind_score = ( vex.jumpkind == b.vex.jumpkind ) ? 1.0 : 0.0;
            assert_score( vex_jumpkind_score );
            similarities.push( vex_jumpkind_score );
            
            //////////////////////////////////////////////////////
	    //sum_operations set intersection
            //////////////////////////////////////////////////////
            var a_sum_ops = [];
            for(arr in vex.sum_operations){
                a_sum_ops.push(arr); //extract keys into array
            }

            var b_sum_ops = [];
            for(arr in b.vex.sum_operations){
                b_sum_ops.push(arr); //extract keys into array
            }

            var aset = new Set( a_sum_ops );
            var bset = new Set( b_sum_ops );

            var sim_ops = new Set();

            var max_ops = Math.max( a_sum_ops.length, b_sum_ops.length );
            if( max_ops > 0 ){
                sim_ops = aset.intersection(bset);
                var sum_matched_ops_score = sim_ops.size / max_ops; //max score 1
                assert_score( sum_matched_ops_score );
                similarities.push( sum_matched_ops_score );
            }else{
                similarities.push( 1.0 );
            }

            //size of both total number of types of operations
            var sum_ops_size_diff_score = abs_diff( aset.size, bset.size, 2.0);
            assert_score( sum_ops_size_diff_score );
            similarities.push( sum_ops_size_diff_score );
	 
            var sum_ops_matched_number_score = 0.0;
	    console.log(sim_ops);
            if(sim_ops.size > 0){
                //match the number of matched type of ops
                // for(op in sim_ops.keys()){
		sim_ops.forEach( function(key, value, s){
			console.log( vex.sum_operations[key] + "\t:\t" + b.vex.sum_operations[key]);
			console.log( abs_diff( 2, 1, 1.6) );
                    console.log( abs_diff( vex.sum_operations[key], b.vex.sum_operations[key], 1.6) );
                    sum_ops_matched_number_score += abs_diff( vex.sum_operations[key], b.vex.sum_operations[key], 1.6);
                });
		    console.log( sum_ops_size_diff_score );
		    console.log(sim_ops.size);
                sum_ops_matched_number_score /= sim_ops.size;
            }else{
                if( max_ops > 0){
                    sum_ops_size_diff_score = 0.0;
                }else{
                    sum_ops_size_diff_score = 1.0;
                }
            }
            console.log("SUM OPS MATCHED SCORE: " +  sum_ops_size_diff_score );
            assert_score( sum_ops_size_diff_score );
            similarities.push( sum_ops_size_diff_score );
            
            //////////////////////////////////////////////////////
	    //sum_statements set intersection
            //////////////////////////////////////////////////////
            var a_sum_stmts = [];
            for(arr in vex.sum_statements){
                a_sum_stmts.push(arr); //extract keys into array
            }

            var b_sum_stmts = [];
            for(arr in b.vex.sum_statements){
                b_sum_stmts.push(arr); //extract keys into array
            }

            var aset = new Set( a_sum_stmts );
            var bset = new Set( b_sum_stmts );
            var sim_stmts = new Set();

            var max_stmts = Math.max( a_sum_stmts.length, b_sum_stmts.length );
            if( max_stmts > 0){
                sim_stmts = aset.intersection(bset);
                var sum_matched_stmts_score = sim_stmts.size / max_stmts; //max score 1
                assert_score( sum_matched_stmts_score );
                similarities.push( sum_matched_stmts_score );
            }else{
                similarities.push( 1.0 );
            }


            //size of both total number of types of statements
            var sum_stmts_size_diff_score = abs_diff( aset.size, bset.size, 2.0);
            assert_score( sum_stmts_size_diff_score );
            similarities.push( sum_stmts_size_diff_score );
	 
            var sum_stmts_matched_number_score = 0.0;
            if(sim_stmts.size > 0){
                //match the number of matched type of stmts
		sim_stmts.forEach( function( key, value, s){
                //for(stmt in sim_stmts.keys()){
                    sum_stmts_matched_number_score += abs_diff( vex.sum_statements[key], b.vex.sum_statements[key], 1.6);
                });
                sum_stmts_matched_number_score /= sim_stmts.size;
            } else {
                if( max_stmts > 0 ){
                    //A or B has sum statements but non similar
                    sum_stmts_matched_number_score = 0.0;
                }else{
                    //A and B have no sum_statements -> therefore similar
                    sum_stmts_matched_number_score = 1.0;
                }
            }
            //console.log( "SUM STMTS MATCHED SCORE: " + sum_stmts_matched_number_score );
            assert_score( sum_stmts_matched_number_score );
            similarities.push( sum_stmts_matched_number_score );
	

            //////////////////////////////////////////////////////
	    //sum_expressions set intersection
            //////////////////////////////////////////////////////
            var a_sum_exprs = [];
            for(arr in vex.sum_expressions){
                a_sum_exprs.push(arr); //extract keys into array
            }

            var b_sum_exprs = [];
            for(arr in b.vex.sum_expressions){
                b_sum_exprs.push(arr); //extract keys into array
            }

            var aset = new Set( a_sum_exprs );
            var bset = new Set( b_sum_exprs );
            var sim_exprs = new Set();

            var max_exprs = Math.max( a_sum_exprs.length, b_sum_exprs.length );
            if(max_exprs > 0){
                sim_exprs = aset.intersection(bset);
                var sum_matched_exprs_score = sim_exprs.size / max_exprs; //max score 1
                assert_score( sum_matched_exprs_score );
                similarities.push( sum_matched_exprs_score );
            }else{
                similarities.push(1.0);
            }

            //size of both total number of types of expressions
            var sum_exprs_size_diff_score = abs_diff( aset.size, bset.size, 2.0);
            assert_score( sum_exprs_size_diff_score );
            similarities.push( sum_exprs_size_diff_score );
	 
            var sum_exprs_matched_number_score = 0.0;
            if( sim_exprs.size > 0){
	    //console.log("SIM EXPRESSIONS");
	    //console.log(sim_exprs);
	    //console.log("A_EXPRS");
	    //console.log(vex.sum_expressions);
	    //console.log("B_EXPRS");
	    //console.log(b.vex.sum_expressions);
                //match the number of matched type of exprs
                //for(expr in sim_exprs.values()){
		sim_exprs.forEach( function( key, value, s){
		    //console.log(vex.sum_expressions[key]);
		    //console.log(b.vex.sum_expressions[key]);
		    //console.log(abs_diff( vex.sum_expressions[key], b.vex.sum_expressions[key], 1.6));
                    sum_exprs_matched_number_score += abs_diff( vex.sum_expressions[key], b.vex.sum_expressions[key], 1.6);
                });
		console.log("Total score: " + sum_exprs_matched_number_score);
                //console.log( sim_exprs.size );

                sum_exprs_matched_number_score /= sim_exprs.size;
            } else {
                if( max_exprs > 0){
                    sum_exprs_matched_number_score = 0.0;
                }else{
                    sum_exprs_matched_number_score = 1.0;
                }
            }
            assert_score( sum_exprs_matched_number_score );
	    console.log("SUM EXPRS MATCHED SCORE: " + sum_exprs_matched_number_score);
            similarities.push( sum_exprs_matched_number_score );


            //match constants

            //match Int constants
            var a_int_consts = [];
            var b_int_consts = [];
            for( const_t in vex.constants){
                //int type
                if( vex.constants[const_t]['type'].match('^Ity') ){
                    a_int_consts.push( Number( vex.constants[const_t]['value'] ) );
                }

                //str type
                //TODO: Not used in pyvex
            }

            for( const_t in b.vex.constants){
                //int type
                if( b.vex.constants[const_t]['type'].match('^Ity') ){
                    b_int_consts.push( Number( b.vex.constants[const_t]['value'] ) );
                }
            }

            var a_consts = new Set( a_int_consts );
            var b_consts = new Set( b_int_consts );

            //unbounded, matching constants is important
            //TODO: bound this score
            var max_consts = Math.max( a_consts.size, b_consts.size );
            if(max_consts > 0){
                var constant_matching_score = a_consts.intersection(b_consts).size / max_consts;
                assert_score( constant_matching_score );
                similarities.push( constant_matching_score );
            }else{
                similarities.push( 1.0 ); //constants match -> they both have none
            }

            //TODO order of operations, statements and expressions
        }

        return similarities;
    }

        var best_symb = null;
        var high_score = -1.0;
        values.forEach(function(symb){
           var score = similarity( symb );
           var fscore = score.reduce( function(a,b){ return a+b;}, 0.0);
           if ( fscore > high_score ){   
               high_score = fscore;
               best_symb = symb;
               best_symb.score = fscore;
               best_symb.similarity = score;
           }
        });
        if( typeof(best_symb) === null ){
            throw 'Error, best_symb not set from ' + values.length + ' symbols';
        }
        best_symb['sum'] = values.reduce( function(sum, t){
            return sum + t.sum;
        }, 0);
        return best_symb;
        }
