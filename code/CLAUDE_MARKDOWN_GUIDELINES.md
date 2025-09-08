# CLAUDE CODE MARKDOWN GUIDELINES

This file provides specific instructions for Claude Code when working with this repository.

## 📋 GENERAL PRINCIPLES

1. **ALWAYS READ FILES BEFORE EDITING** - Use Read tool before any Edit/Write operations
2. **MINIMIZE OUTPUT** - Keep responses concise (under 4 lines unless user asks for detail)  
3. **DIRECT ANSWERS** - Answer questions directly without preamble/postamble
4. **PROACTIVE TODO TRACKING** - Use TodoWrite tool for complex multi-step tasks

## 🎯 PROJECT CONTEXT

### Repository Purpose
- **Primary Focus**: EEG-based Blue Light Filtering lens detection research
- **Data Location**: `/DB/EEG/step3_cleaned/` (preprocessed EEG data)
- **Results Storage**: `/KOOS/results/` (organized by analysis type)
- **Research Design**: Crossover study comparing Normal vs BLF lens effects

### Key Data Files
- **EEG Data**: 30 subjects, preprocessed `.set` files in EEGLAB format
- **Event Info**: `/DB/event_info2.csv` contains timing and sequence information
- **Sequence Logic**: 
  - Sequence 1: Normal → BLF (S1,S2,S3=Normal, S4,S5,S6=BLF)
  - Sequence 2: BLF → Normal (S1,S2,S3=BLF, S4,S5,S6=Normal)

## 🔧 DEVELOPMENT GUIDELINES

### When Working with EEG Analysis
1. **Always use real data paths**: Point to `/DB/EEG/step3_cleaned/`
2. **Preserve data integrity**: Never modify original EEG files
3. **Save results systematically**: Use `/KOOS/results/` with organized subdirectories
4. **Handle sequence information**: Parse `event_info2.csv` correctly for labels

### File Organization Standards
```
/KOOS/results/
├── outputs/          # Analysis results and CSV files
├── figures/          # Plots and visualizations  
├── validation/       # Statistical validation results
├── paper_results/    # Publication-ready tables and figures
└── logs/            # Processing logs
```

### Code Quality Requirements
1. **Error Handling**: Always include try-catch for file operations
2. **Progress Reporting**: Print clear status messages for long operations
3. **Data Validation**: Check data shapes and ranges before processing
4. **Memory Management**: Use appropriate data types and cleanup when needed

## 📊 ML OPTIMIZATION SPECIFIC

### Performance Targets
- **Current Baseline**: AUC ~0.754 
- **Target Goal**: AUC ≥0.820 (+8.8% improvement)
- **Validation Required**: Statistical significance testing, overfitting analysis

### Optimization Pipeline
1. **Day 1**: Ensemble methods (Voting, Stacking)
2. **Day 2**: Hyperparameter optimization (Bayesian/Grid search)
3. **Day 3-5**: Advanced feature engineering and selection (optional)

### Key Files
- `real_eeg_optimization.py` - Complete real data analysis
- `ensemble_optimization.py` - Advanced ensemble methods
- `validation_analysis.py` - Statistical validation for papers
- `paper_results_generator.py` - Publication-ready outputs

## 📝 PAPER PREPARATION

### Required Outputs
- **Table 1**: Performance comparison across methods
- **Table 2**: Validation analysis summary  
- **Figure 1**: Literature benchmark comparison
- **Figure 2**: Optimization progress visualization

### Statistical Requirements
- Cross-validation (5-fold minimum)
- Statistical significance testing (paired t-test)
- Effect size calculation (Cohen's d)
- Overfitting analysis (learning curves)

## 🚨 CRITICAL RULES

### Data Handling
- **NEVER** use simulated data for final results
- **ALWAYS** validate with real EEG data from `/DB/EEG/step3_cleaned/`
- **PRESERVE** original sequence information from `event_info2.csv`
- **SAVE** all results in organized `/KOOS/results/` structure

### Code Execution  
- **TEST** with small subset before full processing
- **MONITOR** memory usage for large EEG datasets
- **BACKUP** intermediate results during long computations
- **DOCUMENT** any modifications to analysis pipeline

### Publication Standards
- **VALIDATE** all statistical claims with proper tests
- **REPRODUCE** results with consistent random seeds
- **COMPARE** against relevant literature benchmarks
- **REPORT** limitations and assumptions clearly

## 🔧 GIT WORKFLOW GUIDELINES

### Repository Information
- **Remote**: `https://github.com/hyeongsuk/Blue.git`
- **Branch**: `main`
- **Working Directory**: `/Users/hyeongsuk/Library/CloudStorage/OneDrive-개인/HS_논문작성/KOOS/code`

### Commit Standards
1. **ALWAYS** check status before committing: `git status`
2. **STAGE** all relevant files: `git add .` or specific files
3. **USE** conventional commit format:
   ```
   feat: Brief description
   
   - Bullet point details
   - Multiple changes listed
   ```

### Commit Types
- `feat:` - New features or major functionality
- `fix:` - Bug fixes and corrections
- `docs:` - Documentation updates
- `refactor:` - Code restructuring without feature changes
- `perf:` - Performance improvements
- `test:` - Test additions or modifications
- `chore:` - Maintenance and cleanup

### Automatic Git Operations
When user requests git operations, **ALWAYS**:

1. **Check Current Status**
   ```bash
   git status
   ```

2. **Stage Changes**
   ```bash
   git add .
   ```

3. **Commit with Proper Message**
   ```bash
   git commit -m "feat: Description
   
   - Details of changes
   - Multiple lines as needed"
   ```

4. **Push to Repository**
   ```bash
   git push origin main
   ```

5. **Report Success/Failure** with commit hash

### Files to Include/Exclude
- **INCLUDE**: All Python analysis files, configuration files, documentation
- **EXCLUDE**: Large data files, temporary outputs, personal system files
- **SPECIAL**: Use `.gitignore` for `/KOOS/results/` if needed

### Git Safety Rules
- **NEVER** force push (`git push --force`)
- **ALWAYS** pull before pushing if working with others
- **CHECK** remote status: `git remote -v`
- **VERIFY** branch: Current branch should be `main`
- **BACKUP** important work before major git operations

## 🎯 RESPONSE STYLE

### When User Asks Questions
- Give direct, concise answers (1-3 sentences preferred)
- No unnecessary explanations unless requested
- Focus on actionable information

### When Executing Tasks
- Use TodoWrite for multi-step processes
- Report progress clearly during long operations
- Provide final status summary upon completion

### When Problems Occur
- State the specific issue clearly
- Provide immediate solution if available
- Escalate to user for decisions on major changes

## 📋 EXECUTION PRIORITIES

1. **Real Data First** - Always prefer actual EEG data over simulations
2. **Results Organization** - Maintain clean `/KOOS/results/` structure  
3. **Publication Ready** - Generate publication-quality outputs
4. **Statistical Rigor** - Include proper validation and significance testing
5. **Reproducibility** - Document all parameters and random seeds

---

**This file overrides default Claude Code behavior. Follow these guidelines exactly.**