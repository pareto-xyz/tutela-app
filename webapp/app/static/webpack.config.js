const webpack = require('webpack');
const config = {
<<<<<<< HEAD
<<<<<<< HEAD
    entry: __dirname + '/js/index.js',
=======
    entry:  __dirname + '/js/index.js',
>>>>>>> b4ef37e (add react (unfinished))
=======
    entry:  __dirname + '/js/index.js',
>>>>>>> b4ef37e (add react (unfinished))
    output: {
        path: __dirname + '/dist',
        filename: 'bundle.js',
    },
    resolve: {
        extensions: ['.js', '.jsx', '.css']
    },
<<<<<<< HEAD
<<<<<<< HEAD

    module: {
        rules: [
            {
                test: /\.(js|jsx)?/,
                exclude: /node_modules/,
                use: 'babel-loader'
            },
            {
                test: /\.(png|svg|jpg|gif)$/,
                use: 'file-loader'
            },
            {
                test: /\.json$/,
                loader: 'json-loader'
            }
        ],
=======
=======
>>>>>>> b4ef37e (add react (unfinished))
  
    module: {
        rules: [
			{
			test: /\.(js|jsx)?/,
				exclude: /node_modules/,
				use: 'babel-loader'		
			},
			{
				test: /\.(png|svg|jpg|gif)$/,
				use: 'file-loader'
			}			
		]
<<<<<<< HEAD
>>>>>>> b4ef37e (add react (unfinished))
=======
>>>>>>> b4ef37e (add react (unfinished))
    }
};
module.exports = config;