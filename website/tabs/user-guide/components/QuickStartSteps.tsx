import React from 'react';

interface Step {
  title: string;
  description: string;
  code?: string;
}

interface QuickStartStepsProps {
  steps: Step[];
}

export default function QuickStartSteps({ steps }: QuickStartStepsProps): JSX.Element {
  return (
    <div className="quickstart-steps">
      {steps.map((step, index) => (
        <div key={index} className="quickstart-step">
          <div className="quickstart-step-number">{index + 1}</div>
          <div className="quickstart-step-content">
            <h4 className="quickstart-step-title">{step.title}</h4>
            <p className="quickstart-step-description">{step.description}</p>
            {step.code && (
              <pre className="quickstart-step-code">
                <code>{step.code}</code>
              </pre>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
