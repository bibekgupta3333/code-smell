module.exports = [
  {
    ignores: ['node_modules/', '.venv/', '**/dist/**', '**/build/**']
  },
  {
    files: ['**/*.js', '**/*.jsx'],
    languageOptions: {
      ecmaVersion: 'latest',
      sourceType: 'module',
      globals: {
        console: 'readonly',
        process: 'readonly',
        Buffer: 'readonly',
        __dirname: 'readonly',
        __filename: 'readonly',
        global: 'readonly',
        setImmediate: 'readonly',
        setInterval: 'readonly',
        setTimeout: 'readonly'
      }
    },
    rules: {
      'no-unused-vars': ['warn', { argsIgnorePattern: '^_' }],
      'no-console': 'off',
      'semi': ['warn', 'always'],
      'quotes': ['warn', 'single', { avoidEscape: true }],
      'indent': ['warn', 2],
      'no-var': 'warn',
      'prefer-const': 'warn',
      'eqeqeq': ['warn', 'always'],
      'no-implied-eval': 'warn',
      'no-with': 'warn',
      'no-eval': 'warn',
      'no-multiple-empty-lines': ['warn', { max: 2 }],
      'no-empty-function': 'warn',
      'no-empty': 'warn',
      'no-duplicate-imports': 'warn',
      'no-dupe-keys': 'warn'
    }
  }
];
