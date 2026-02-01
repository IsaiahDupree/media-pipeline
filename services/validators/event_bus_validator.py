"""Event bus validation"""
from services.validation_framework import ValidationResult, ValidationSeverity
from services.event_bus import EventBus


async def validate_event_bus() -> ValidationResult:
    """Validate event bus connectivity"""
    result = ValidationResult(component="event_bus")
    
    try:
        event_bus = EventBus.get_instance()
        result.metadata["event_bus_type"] = type(event_bus).__name__
        
        # Test publishing an event
        try:
            await event_bus.publish(
                "validation.test",
                {"test": True},
                correlation_id="validation-test"
            )
            result.metadata["can_publish"] = True
        except Exception as e:
            result.add_issue(
                ValidationSeverity.WARNING,
                f"Cannot publish events: {str(e)}",
                details="Event bus may not be fully functional"
            )
            result.metadata["can_publish"] = False
    
    except Exception as e:
        result.add_issue(
            ValidationSeverity.CRITICAL,
            f"Event bus initialization failed: {str(e)}",
            details=str(e)
        )
    
    return result

