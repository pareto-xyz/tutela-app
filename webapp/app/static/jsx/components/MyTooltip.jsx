import React from 'react';
import { OverlayTrigger, Tooltip } from 'react-bootstrap';

export default function MyTooltip({tooltipText}) {

    const renderHelpTooltip = props => {
        return (
            <Tooltip {...props} className="tooltip">
                {tooltipText}
            </Tooltip>
        );
    }

    return (
        <OverlayTrigger
            placement="right"
            delay={{ show: 250, hide: 400 }}
            overlay={renderHelpTooltip}
        >
            <div className="help-circle"><i className="far fa-question-circle"></i></div>
        </OverlayTrigger>
    );
}