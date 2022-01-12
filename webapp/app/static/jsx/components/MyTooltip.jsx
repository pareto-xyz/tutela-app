import React from 'react';
import { OverlayTrigger, Tooltip } from 'react-bootstrap';

export default function MyTooltip({ tooltipText }) {

    const renderHelpTooltip = props => {
        return (
            <Tooltip {...props} className="tooltip">
                {tooltipText}
            </Tooltip>
        );
    }

    return (
        <OverlayTrigger
            placement="bottom"
            delay={{ show: 250, hide: 400 }}
            overlay={renderHelpTooltip}
        >
            <i className="col-2 far fa-question-circle"></i>
        </OverlayTrigger>
    );
}