# Repository Configuration Guide

This document provides the recommended repository settings for the Spin-Torque RL-Gym project to ensure security, quality, and compliance with SDLC best practices.

## General Settings

### Repository Details
- **Description**: Gymnasium environment for spin-torque device control via reinforcement learning
- **Website**: `https://spinrl.com` (when available)
- **Topics**: 
  - `reinforcement-learning`
  - `gymnasium`
  - `spin-torque`
  - `magnetic-devices`
  - `machine-learning`
  - `physics-simulation`
  - `python`
  - `research`

### Features
- ✅ **Issues**: Enable issue tracking
- ✅ **Projects**: Enable project boards
- ✅ **Wiki**: Enable wiki (for extended documentation)
- ✅ **Discussions**: Enable discussions for community interaction
- ✅ **Sponsorships**: Enable if accepting sponsorships

### Pull Requests
- ✅ **Allow merge commits**: Enable
- ✅ **Allow squash merging**: Enable (default)
- ✅ **Allow rebase merging**: Enable
- ✅ **Always suggest updating pull request branches**: Enable
- ✅ **Allow auto-merge**: Enable
- ✅ **Automatically delete head branches**: Enable

## Security Settings

### Code Security and Analysis

#### Dependency Graph
- ✅ **Enable dependency graph**: Required for security alerts

#### Dependabot Alerts
- ✅ **Enable Dependabot alerts**: Monitor for vulnerabilities
- ✅ **Enable Dependabot security updates**: Auto-create PRs for security fixes

#### Code Scanning
- ✅ **Enable CodeQL analysis**: Set up via `.github/workflows/security-scan.yml`
- **Query suites**: Use `security-and-quality` and `security-extended`

#### Secret Scanning
- ✅ **Enable secret scanning**: GitHub automatically scans for secrets
- ✅ **Enable push protection**: Prevent commits containing secrets

### Access and Permissions

#### Collaborators and Teams
Create the following teams with appropriate access levels:

1. **@maintainer** (Admin access)
   - Repository administration
   - All permissions

2. **@core-developers** (Write access)
   - Code review and development
   - Can merge PRs after review

3. **@devops-team** (Write access)
   - CI/CD and infrastructure
   - Docker and deployment files

4. **@security-team** (Write access)
   - Security-related files and configurations
   - Security scan review

5. **@qa-team** (Write access)
   - Testing infrastructure
   - Test review and quality assurance

6. **@technical-writers** (Write access)
   - Documentation
   - README and docs maintenance

7. **@research-team** (Read access)
   - Research collaboration
   - Can create issues and discussions

8. **@community-team** (Triage access)
   - Community management
   - Issue triage and community guidelines

## Branch Protection Rules

### Main Branch (`main`)

#### Protect matching branches
- ✅ **Branch name pattern**: `main`

#### Restrictions
- ✅ **Restrict pushes that create files**
- ❌ **Allow force pushes**: Disabled
- ❌ **Allow deletions**: Disabled

#### Rules

##### Pull Request Requirements
- ✅ **Require a pull request before merging**
- ✅ **Require approvals**: 1 approval minimum
- ✅ **Dismiss stale PR approvals when new commits are pushed**
- ✅ **Require review from CODEOWNERS**
- ✅ **Restrict pushes that create files**

##### Status Check Requirements
- ✅ **Require status checks to pass before merging**
- ✅ **Require branches to be up to date before merging**

**Required Status Checks**:
- `Test Suite (ubuntu-latest, 3.11)`
- `Code Quality`
- `Build Package`
- `All Checks Complete`
- `Build Docker Images (production)`

##### Additional Restrictions
- ✅ **Require conversation resolution before merging**
- ✅ **Require signed commits** (recommended for security)
- ✅ **Include administrators**: Apply rules to repository administrators

### Development Branch (`develop`)

#### Protect matching branches
- ✅ **Branch name pattern**: `develop`

#### Rules
- ✅ **Require a pull request before merging**
- ✅ **Require approvals**: 1 approval minimum
- ✅ **Require status checks to pass before merging**

**Required Status Checks**:
- `Test Suite (ubuntu-latest, 3.11)`
- `Code Quality`

## Environments

### Staging Environment
- **Name**: `staging`
- **Protection Rules**:
  - ❌ **Required reviewers**: None (automatic deployment)
  - ⏰ **Wait timer**: 0 minutes
  - 🌿 **Deployment branches**: `main` and `develop` branches only

**Environment Secrets**:
- `STAGING_URL`: Staging application URL
- `STAGING_DB_URL`: Staging database connection string
- `STAGING_DEPLOY_KEY`: SSH key for staging deployment

### Production Environment
- **Name**: `production`
- **Protection Rules**:
  - ✅ **Required reviewers**: @maintainer, @devops-team (minimum 1)
  - ⏰ **Wait timer**: 5 minutes
  - 🌿 **Deployment branches**: `main` branch and release tags only

**Environment Secrets**:
- `PRODUCTION_URL`: Production application URL
- `PRODUCTION_DB_URL`: Production database connection string
- `PRODUCTION_DEPLOY_KEY`: SSH key for production deployment

## Repository Secrets

### Required Secrets

#### GitHub Integration
- `GITHUB_TOKEN`: Automatically provided by GitHub Actions

#### Deployment
- `STAGING_DEPLOY_KEY`: SSH private key for staging deployment
- `PRODUCTION_DEPLOY_KEY`: SSH private key for production deployment

#### Notifications
- `SLACK_WEBHOOK`: Webhook URL for Slack notifications
- `SECURITY_SLACK_WEBHOOK`: Dedicated webhook for security alerts

#### External Services
- `CODECOV_TOKEN`: Token for Codecov integration (optional)
- `SNYK_TOKEN`: Token for Snyk security scanning (optional)

### Variables

#### Environment URLs
- `STAGING_URL`: `https://staging.spinrl.com`
- `PRODUCTION_URL`: `https://spinrl.com`

#### Configuration
- `DOCKER_REGISTRY`: `ghcr.io`
- `IMAGE_NAME`: `danieleschmidt/spin-torque-rl-gym`

## Issue and PR Templates

### Issue Templates

1. **Bug Report** (`.github/ISSUE_TEMPLATE/bug_report.yml`)
2. **Feature Request** (`.github/ISSUE_TEMPLATE/feature_request.yml`)
3. **Research Question** (`.github/ISSUE_TEMPLATE/research_question.yml`)
4. **Documentation Improvement** (`.github/ISSUE_TEMPLATE/documentation.yml`)

### Pull Request Template
- Location: `.github/PULL_REQUEST_TEMPLATE.md`
- Includes checklists for testing, documentation, and code quality

## Labels

### Bug Labels
- `bug` (🐛): Something isn't working
- `critical` (🚨): Critical issue requiring immediate attention
- `regression` (⏪): Previously working functionality is broken

### Enhancement Labels
- `enhancement` (✨): New feature or request
- `performance` (🚀): Performance improvement
- `refactoring` (🔧): Code improvement without functionality changes

### Process Labels
- `needs-triage` (🔍): Needs initial review and categorization
- `needs-review` (👀): Awaiting code review
- `needs-testing` (🧪): Requires testing before merge
- `ready-to-merge` (✅): Approved and ready for merge

### Area Labels
- `area/core` (⚙️): Core environment functionality
- `area/physics` (🔬): Physics simulation components
- `area/docs` (📖): Documentation
- `area/ci-cd` (🔄): Continuous integration and deployment
- `area/security` (🔒): Security-related issues

### Priority Labels
- `priority/low` (🟢): Low priority
- `priority/medium` (🟡): Medium priority
- `priority/high` (🟠): High priority
- `priority/critical` (🔴): Critical priority

## Webhooks and Integrations

### Recommended Integrations
1. **Codecov**: Code coverage reporting
2. **Dependabot**: Automated dependency updates
3. **Slack**: Team notifications
4. **CodeQL**: Security analysis

### Webhook Events
Configure webhooks for the following events:
- Push events (for CI/CD triggers)
- Pull request events (for code review automation)
- Issue events (for project management)
- Release events (for deployment automation)

## Compliance and Governance

### SLSA Compliance
- Implement SLSA Level 3 requirements
- Ensure build provenance and integrity
- Use signed container images

### Audit Requirements
- Enable audit log retention
- Configure security event monitoring
- Implement compliance reporting

### Data Protection
- Follow data minimization principles
- Implement secure secret management
- Regular security assessments

## Monitoring and Metrics

### Repository Health Metrics
- Track commit frequency and patterns
- Monitor pull request review times
- Measure issue resolution times
- Security vulnerability response times

### Quality Metrics
- Code coverage percentage
- Test success rates
- Build success rates
- Security scan results

### Community Metrics
- Community contributions percentage
- Discussion activity levels
- Documentation usage analytics

## Implementation Checklist

### Initial Setup
- [ ] Configure repository description and topics
- [ ] Set up branch protection rules
- [ ] Create teams and assign permissions
- [ ] Configure environments and secrets
- [ ] Set up issue and PR templates

### Security Configuration
- [ ] Enable dependency scanning
- [ ] Configure CodeQL analysis
- [ ] Set up secret scanning
- [ ] Configure signed commits requirement

### CI/CD Setup
- [ ] Copy workflow templates to `.github/workflows/`
- [ ] Configure required status checks
- [ ] Set up deployment environments
- [ ] Test CI/CD pipeline

### Community Setup
- [ ] Create contribution guidelines
- [ ] Set up issue templates
- [ ] Configure discussion categories
- [ ] Establish code of conduct

### Monitoring Setup
- [ ] Configure webhook integrations
- [ ] Set up notification channels
- [ ] Implement metrics collection
- [ ] Create monitoring dashboards

This configuration ensures a secure, scalable, and maintainable repository that follows industry best practices for software development lifecycle management.