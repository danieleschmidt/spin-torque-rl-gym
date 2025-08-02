# Manual Setup Required

Due to GitHub App permission limitations, certain repository configurations must be set up manually by repository maintainers.

## Workflow Files Setup

### Step 1: Copy Workflow Templates

Copy the workflow templates from the examples directory to the `.github/workflows/` directory:

```bash
# Create workflows directory
mkdir -p .github/workflows

# Copy workflow files
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/cd.yml .github/workflows/
cp docs/workflows/examples/security-scan.yml .github/workflows/
cp docs/workflows/examples/dependency-update.yml .github/workflows/
```

### Step 2: Configure Repository Secrets

Add the following secrets in **Repository Settings > Secrets and Variables > Actions**:

#### Required Secrets

| Secret Name | Description | Example Value |
|-------------|-------------|---------------|
| `GITHUB_TOKEN` | Automatically provided | (automatic) |

#### Deployment Secrets

| Secret Name | Description | Example Value |
|-------------|-------------|---------------|
| `STAGING_URL` | Staging environment URL | `https://staging.spinrl.com` |
| `PRODUCTION_URL` | Production environment URL | `https://spinrl.com` |
| `STAGING_DEPLOY_KEY` | SSH key for staging deployment | `-----BEGIN OPENSSH PRIVATE KEY-----...` |
| `PRODUCTION_DEPLOY_KEY` | SSH key for production deployment | `-----BEGIN OPENSSH PRIVATE KEY-----...` |

#### Notification Secrets

| Secret Name | Description | Example Value |
|-------------|-------------|---------------|
| `SLACK_WEBHOOK` | Slack webhook for notifications | `https://hooks.slack.com/services/...` |
| `SECURITY_SLACK_WEBHOOK` | Slack webhook for security alerts | `https://hooks.slack.com/services/...` |

#### Optional Enhancement Secrets

| Secret Name | Description | Example Value |
|-------------|-------------|---------------|
| `CODECOV_TOKEN` | Codecov token for coverage reports | `12345678-1234-1234-1234-123456789012` |
| `SNYK_TOKEN` | Snyk token for vulnerability scanning | `12345678-1234-1234-1234-123456789012` |
| `GITLEAKS_LICENSE` | GitLeaks license key | `12345678-1234-1234-1234-123456789012` |

### Step 3: Configure Repository Variables

Add the following variables in **Repository Settings > Secrets and Variables > Actions > Variables**:

| Variable Name | Description | Example Value |
|---------------|-------------|---------------|
| `STAGING_URL` | Staging environment URL | `https://staging.spinrl.com` |
| `PRODUCTION_URL` | Production environment URL | `https://spinrl.com` |

### Step 4: Set Up Environments

Create environments in **Repository Settings > Environments**:

#### Staging Environment
- **Name**: `staging`
- **Deployment branches**: `main` branch only
- **Environment secrets**:
  - `STAGING_URL`: Staging application URL
  - `STAGING_DB_URL`: Staging database URL
- **Protection rules**: None (automatic deployment)

#### Production Environment
- **Name**: `production`
- **Deployment branches**: `main` branch and release tags
- **Environment secrets**:
  - `PRODUCTION_URL`: Production application URL
  - `PRODUCTION_DB_URL`: Production database URL
- **Protection rules**:
  - ✅ Required reviewers: Add team leads/maintainers
  - ✅ Wait timer: 5 minutes
  - ⏰ Deployment branches rule: `main` and release tags only

### Step 5: Configure Branch Protection

Set up branch protection for the `main` branch in **Repository Settings > Branches**:

#### Protection Rules
- ✅ **Require a pull request before merging**
  - ✅ Require approvals: 1
  - ✅ Dismiss stale PR approvals when new commits are pushed
  - ✅ Require review from CODEOWNERS
- ✅ **Require status checks to pass before merging**
  - ✅ Require branches to be up to date before merging
  - **Required status checks**:
    - `Test Suite (ubuntu-latest, 3.11)`
    - `Code Quality`
    - `Build Package`
    - `All Checks Complete`
- ✅ **Require conversation resolution before merging**
- ✅ **Require signed commits**
- ✅ **Restrict pushes that create files**
- ❌ **Allow force pushes** (disabled)
- ❌ **Allow deletions** (disabled)

### Step 6: Set Up Dependabot

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "08:00"
      timezone: "UTC"
    open-pull-requests-limit: 10
    reviewers:
      - "team-leads"
    assignees:
      - "maintainer"
    commit-message:
      prefix: "⬆️"
      include: "scope"
    labels:
      - "dependencies"
      - "python"
    
  # Docker dependencies
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
      timezone: "UTC"
    reviewers:
      - "team-leads"
    commit-message:
      prefix: "⬆️"
      include: "scope"
    labels:
      - "dependencies"
      - "docker"
  
  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "10:00"
      timezone: "UTC"
    commit-message:
      prefix: "⬆️"
      include: "scope"
    labels:
      - "dependencies"
      - "github-actions"
```

### Step 7: Configure Issue Templates

Create `.github/ISSUE_TEMPLATE/`:

```bash
mkdir -p .github/ISSUE_TEMPLATE
```

**Bug Report** (`.github/ISSUE_TEMPLATE/bug_report.yml`):
```yaml
name: Bug Report
description: File a bug report to help us improve
title: "[Bug]: "
labels: ["bug", "needs-triage"]
assignees: []
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to fill out this bug report!
  
  - type: textarea
    id: what-happened
    attributes:
      label: What happened?
      description: A clear description of the bug
      placeholder: Tell us what you see!
    validations:
      required: true
  
  - type: textarea
    id: expected
    attributes:
      label: Expected behavior
      description: What did you expect to happen?
    validations:
      required: true
  
  - type: textarea
    id: steps
    attributes:
      label: Steps to reproduce
      description: How can we reproduce this issue?
      placeholder: |
        1. Go to '...'
        2. Click on '....'
        3. Scroll down to '....'
        4. See error
    validations:
      required: true
  
  - type: textarea
    id: environment
    attributes:
      label: Environment
      description: |
        Please provide details about your environment:
      value: |
        - OS: [e.g. Ubuntu 20.04, macOS 12.0, Windows 10]
        - Python version: [e.g. 3.11.0]
        - Package version: [e.g. 0.1.0]
    validations:
      required: true
  
  - type: textarea
    id: logs
    attributes:
      label: Relevant log output
      description: Please copy and paste any relevant log output
      render: shell
```

**Feature Request** (`.github/ISSUE_TEMPLATE/feature_request.yml`):
```yaml
name: Feature Request
description: Suggest an idea for this project
title: "[Feature]: "
labels: ["enhancement", "needs-triage"]
assignees: []
body:
  - type: markdown
    attributes:
      value: |
        Thanks for suggesting a new feature!
  
  - type: textarea
    id: problem
    attributes:
      label: Is your feature request related to a problem?
      description: A clear description of what the problem is
      placeholder: I'm always frustrated when [...]
  
  - type: textarea
    id: solution
    attributes:
      label: Describe the solution you'd like
      description: A clear description of what you want to happen
    validations:
      required: true
  
  - type: textarea
    id: alternatives
    attributes:
      label: Describe alternatives you've considered
      description: Any alternative solutions or features you've considered
  
  - type: textarea
    id: additional-context
    attributes:
      label: Additional context
      description: Add any other context or screenshots about the feature request
```

### Step 8: Configure Pull Request Template

Create `.github/PULL_REQUEST_TEMPLATE.md`:

```markdown
## Description

Brief description of the changes made in this PR.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance impact assessed

## Checklist

- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes

## Related Issues

Closes #(issue number)

## Screenshots (if applicable)

Add screenshots to help explain your changes.

## Additional Notes

Any additional information that reviewers should be aware of.
```

### Step 9: Create CODEOWNERS File

Create `.github/CODEOWNERS`:

```
# Global code owners
* @maintainer @team-lead

# Workflow and CI/CD
/.github/ @devops-team @maintainer

# Documentation
/docs/ @technical-writer @maintainer

# Security-related files
/docker/ @security-team @devops-team
Dockerfile @security-team @devops-team
*.yml @devops-team

# Python core
/spin_torque_gym/ @core-developers @maintainer

# Tests
/tests/ @qa-team @core-developers

# Configuration files
pyproject.toml @maintainer
requirements*.txt @maintainer
```

### Step 10: Configure CodeQL

Create `.github/codeql-config.yml`:

```yaml
name: "CodeQL Config"

queries:
  - uses: security-and-quality
  - uses: security-extended

paths-ignore:
  - "tests/"
  - "docs/"
  - "scripts/"

paths:
  - "spin_torque_gym/"

disable-default-queries: false
```

## Validation Checklist

After completing the setup, verify the following:

### Workflow Validation
- [ ] All workflow files are in `.github/workflows/`
- [ ] CI workflow triggers on pull requests
- [ ] CD workflow triggers on pushes to main
- [ ] Security scanning runs on schedule
- [ ] Dependency updates run weekly

### Security Configuration
- [ ] All required secrets are configured
- [ ] Environment protection rules are enabled
- [ ] Branch protection is configured
- [ ] CodeQL analysis is enabled
- [ ] Dependabot is configured

### Access Control
- [ ] CODEOWNERS file is configured
- [ ] Team permissions are set correctly
- [ ] Environment reviewers are assigned
- [ ] Branch protection reviewers are required

### Integration Testing
- [ ] Create a test PR to verify CI workflow
- [ ] Test deployment to staging environment
- [ ] Verify security scanning reports
- [ ] Check notification integrations

## Troubleshooting

### Common Issues

**Workflow not triggering**:
- Check file permissions and syntax
- Verify trigger conditions match your setup
- Check if workflow is disabled in repository settings

**Secrets not accessible**:
- Verify secret names match exactly (case-sensitive)
- Check environment-specific secrets are configured
- Ensure secrets are available in the correct scope

**Branch protection not working**:
- Verify status check names match workflow job names
- Check if branch protection applies to the correct branches
- Ensure required reviewers have repository access

**Deployment failures**:
- Verify deployment credentials and endpoints
- Check environment-specific configurations
- Test manual deployment process first

### Getting Help

If you encounter issues during setup:

1. Check the [GitHub Actions documentation](https://docs.github.com/en/actions)
2. Review workflow run logs for specific error messages
3. Verify all prerequisites are met
4. Check repository permissions and access levels

## Security Considerations

- **Never commit secrets** to the repository
- **Use environment-specific secrets** for different deployment targets
- **Regularly rotate secrets** and access tokens
- **Monitor security alerts** and address them promptly
- **Review and audit** access permissions regularly

This setup provides a comprehensive SDLC implementation with security, quality, and compliance built-in. All components work together to ensure reliable, secure, and efficient software delivery.