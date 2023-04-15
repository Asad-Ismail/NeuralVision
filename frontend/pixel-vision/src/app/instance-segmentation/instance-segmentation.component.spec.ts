import { ComponentFixture, TestBed } from '@angular/core/testing';

import { InstanceSegmentationComponent } from './instance-segmentation.component';

describe('InstanceSegmentationComponent', () => {
  let component: InstanceSegmentationComponent;
  let fixture: ComponentFixture<InstanceSegmentationComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ InstanceSegmentationComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(InstanceSegmentationComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
