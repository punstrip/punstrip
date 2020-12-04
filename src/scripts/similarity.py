#!/usr/bin/env python3
from bson.code import Code

#Similarity template based on (size, hash, opcode hash)
similarity_template_simple = """
    var similarity = function(b){
        var score = 0.0;

        //SIZE SIM
        if( {{size}} > 0 ){
            var sharpness = {{sharpness}};
            var size_max = Math.max( {{size}}, b.size );
            var abs_diff = Math.sqrt( Math.pow( {{size}} - b.size, 2 ) )
            var size_sim = ( Math.pow( size_max , sharpness ) - Math.pow( abs_diff, sharpness) )/ Math.pow( size_max, sharpness );
            score += size_sim;
        }

        //HASH SIM
        var hash_sim = (( "{{hash}}" == b.hash) ? 1.0 : 0.0 );
        score += hash_sim;

        //OPCODE HASH SIM
        var opcode_hash_sim = (( "{{opcode_hash}}" == b.opcode_hash) ? 1.0 : 0.0 );
        score += opcode_hash_sim;

        return score;
    }
"""

vector_similarity = """
    var check_dimensions = function(a, b){
        assert(a.length == b.length);
    };
    var dot_product = function(a, b){
        sum = 0.0;
        for(var i=0;i<a.length;i++){
            sum += a[i] * b[i];
        }
        return sum;
    };
    //compute vector l2 norm
    var norm = function(a){
        sum = 0.0;
        for(var i=0; i<a.length;i++){
            sum += Math.pow(a[i], 2);
        }
        return Math.sqrt(sum);
    };
    var angle_between_vectors = function(a, b){
        return Math.acos( dot_product(a, b) / ( norm(a) * norm(b) ) );
    }

    var similarity = function(a, b){
        check_dimensions(a, b);
        return angle_between_vectors(a, b);
    }
"""



similarity_template_vector = """
    var check_dimensions = function(a, b){
        assert(a.length == b.length);
    };
    var dot_product = function(a, b){
        sum = 0.0;
        for(var i=0;i<a.length;i++){
            sum += a[i] * b[i];
        }
        return sum;
    };
    //compute vector l2 norm
    var norm = function(a){
        sum = 0.0;
        for(var i=0; i<a.length;i++){
            sum += Math.pow(a[i], 2);
        }
        return Math.sqrt(sum);
    };
    var angle_between_vectors = function(a, b){
        return Math.acos( dot_product(a, b) / ( norm(a) * norm(b) ) );
    }

    var similarity = function(b){
        var a = {{{vex_vector}}};
        check_dimensions(a, b);
        return angle_between_vectors(a, b);
    }
"""

def gen_map(map_string):
    return Code(map_string)


map_vector_template = Code("""
    function() {
        test_symbs = {{{test_symbs_arr}}};
        for(var i=0; i<test_symbs.length;i++){
            emit(test_symbs[i], { 
                'symbol_id': this.symb_id,
                'vex_vector' : this.vex_vector
            });
        }
    }
""")


map_vector = Code("""
    function() {

        emit(1, { 
            'symbol_id': this.symb_id,
            'vex_vector' : this.vex_vector
        });
    }
""")



map_simple = Code("function() {"
            "emit( 1, { 'size': this.size, 'hash': this.hash, 'opcode_hash': this.opcode_hash, 'id': this._id, 'sum': 1 } );"
            "}")


map_complex = Code("function() {"
        "emit( 1, { 'size': this.size, 'hash': this.hash, 'opcode_hash': this.opcode_hash, 'vex' : this.vex, 'id': this._id, 'sum': 1, 'score': -1.0, 'similarity': [] } );"
    "}")


def gen_reduce_top_three(js_similarity):
    return Code("""function(key, values){
            """ + js_similarity + """
        var best_symb = null;
        var weightings = [ 0.2, 1.0, 1.0, 0.5, 0.7, 0.3, 1, 1, 1, 1, 1, 0.8, 0.9, 0.9, 0.7, 0.3 ];
        var bs = { 'fscore': 0 };
        var best_symbs = [ bs, bs, bs ];
        var min_fscore = 0;

        var high_score = -1.0;
        values.forEach(function(symb){
           var score = similarity( symb );
           var fscore = score.reduce( function(sum, elem, index){ return sum + (elem*weightings[index]);}, 0.0);
           if ( fscore > min_fscore) ){   
               high_score = fscore;
               best_symb = symb;
               best_symb.score = fscore;
               best_symb.similarity = score;

               best_symbs.pop();
               best_symbs.append(best_symb);
               min_fscore = best_symbs.reduce( function(a, b){ return a['fscore'] < b['fscore'] ? a['fscore'] : b['fscore']; });
           }
        });
        if( typeof(best_symb) === null ){
            throw 'Error, best_symb not set from ' + values.length + ' symbols';
        }
        best_symbs['sum'] = values.reduce( function(sum, t){
            return sum + t.sum;
        }, 0);
        return best_symbs;
        }""")


reduce_vector = Code("""function(key, values){
            """ + vector_similarity + """
        var best_symb = null;
        var best_score = 2 * Math.PI; //Max

        values.forEach(function(symb){
            //assert(symb != null);
            if(best_score == 0.0){
                return;
            }
            var score = similarity(key,  symb['vex_vector'] );
            if ( score < best_score ){   
                best_score = score;
                best_symb = symb;
                best_symb.score = score;
            }
        });
        return best_symb;
        }""")



###TODO finish mapping of map-reduce to mulit-key map reduce. Only need 1 map reduce job




def gen_reduce_vector(js_similarity):
    return Code("""function(key, values){
            """ + js_similarity + """
        var best_symb = null;
        var best_score = 2 * Math.PI; //Max

        values.forEach(function(symb){
            //assert(symb != null);
            if(best_score == 0.0){
                return;
            }
            var score = similarity( symb['vex_vector'] );
            if ( score < best_score ){   
                best_score = score;
                best_symb = symb;
                best_symb.score = score;
            }
        });
        return best_symb;
        }""")

def gen_reduce(js_similarity):
    return Code("""function(key, values){
            """ + js_similarity + """
        var best_symb = null;
        var weightings = [ 0.2, 1.0, 1.0, 0.5, 0.7, 0.3, 1, 1, 1, 1, 1, 0.8, 0.9, 0.9, 0.7, 0.3 ];
        var high_score = -1.0;
        values.forEach(function(symb){
           var score = similarity( symb );
           var fscore = score.reduce( function(sum, elem, index){ return sum + (elem*weightings[index]);}, 0.0);
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
        }""")

similarity_template_complex = """
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
        if( {{{size}}} > 0 ){
            var size_score = abs_diff( {{{size}}}, b.size, 0.8);
            assert_score( size_score );
            similarities.push( size_score );
        }else {
            similarities.push( 0.0 );
        }

        //HASH SIM
        var hash_sim = (( "{{{hash}}}" == b.hash) ? 1.0 : 0.0 );
        assert_score( hash_sim );
        similarities.push( hash_sim );

        //OPCODE HASH SIM
        var opcode_hash_sim = (( "{{{opcode_hash}}}" == b.opcode_hash) ? 1.0 : 0.0 );
        assert_score( opcode_hash_sim );
        similarities.push( opcode_hash_sim );

        var vex = {{{vex}}};
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
            if(sim_ops.size > 0){
                //match the number of matched type of ops
		sim_ops.forEach( function(key, value, s){
                    sum_ops_matched_number_score += abs_diff( vex.sum_operations[key], b.vex.sum_operations[key], 1.6);
                });
                sum_ops_matched_number_score /= sim_ops.size;
            }else{
                if( max_ops > 0){
                    sum_ops_size_diff_score = 0.0;
                }else{
                    sum_ops_size_diff_score = 1.0;
                }
            }
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
                //match the number of matched type of exprs
		sim_exprs.forEach( function( key, value, s){
                    sum_exprs_matched_number_score += abs_diff( vex.sum_expressions[key], b.vex.sum_expressions[key], 1.6);
                });

                sum_exprs_matched_number_score /= sim_exprs.size;
            } else {
                if( max_exprs > 0){
                    sum_exprs_matched_number_score = 0.0;
                }else{
                    sum_exprs_matched_number_score = 1.0;
                }
            }
            assert_score( sum_exprs_matched_number_score );
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
            var max_consts = Math.min( vex.constants.length, b.vex.constants.length );
            if(max_consts > 0){
                var constant_matching_score = abs_diff( a_consts.intersection(b_consts).size, max_consts, 0.8);
                assert_score( constant_matching_score );
                similarities.push( constant_matching_score );
            }else{
                similarities.push( 1.0 ); //constants match -> they both have none
            }

            //TODO order of operations, statements and expressions
        }else if( vex == null && b.vex == null ) {
            //both symbols have null vex features -> they are alike
            for(var i = 0; i < 13; ++i){
                similarities.push(1.0);
            }
        }else{
            //1 symbol has vex features, teh other doesn't -> dissimilar
            //push 0 similarity scores
            for(var i = 0; i < 13; ++i){
                similarities.push(0.0);
            }
        }

        return similarities;
    }
"""
